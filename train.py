import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import lavd
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import Batch, Collate, SegmentationDataset  # noqa: F401
from dataset.features import CTC_SPIKE_MODES, DEFAULT_FEATURES, FeatureSelector
from debugger import breakpoint
from dist import sync_dict_values
from ema import AveragedModel
from lr_scheduler import (
    LR_SCHEDULER_KINDS,
    LR_WARMUP_MODES,
    BaseLrScheduler,
    create_lr_scheduler,
)
from model import MODEL_KINDS, BaseSegmentationModel, create_model, unwrap_model
from model.activation import ACTIVATION_KINDS
from model.rnn import RNN_KINDS
from prediction import create_segments
from stats import METRICS, METRICS_DICT, StatsTracker, average_checkpoint_metric
from stats.iou import points_iou
from stats.log import log_epoch_stats, log_experiment, log_results, log_top_checkpoints
from utils import split_named_arg


# This is a class containing all defaults, it is not meant to be instantiated, but
# serves as a sort of const struct.
# It uses nested classes, which also don't follow naming conventions because the idea is
# to have it as a sort of struct. This is kind of like having th defaults defined in
# submodules if modules where first-class constructs, but in one file, because these are
# purely for the training and being able to group various options into one category is
# really nice.
# e.g. DEFAULTS.lr.scheduler accesses the default learning rate scheduler, which is in
# the category lr, where there are various other options regarding the learning rate.
class DEFAULTS:
    seed = 1234
    batch_size = 20
    num_workers = mp.cpu_count()
    num_gpus = torch.cuda.device_count()
    num_epochs = 100

    class features:
        # List of features
        selection = DEFAULT_FEATURES
        ctc_spike_mode = "single"

    class lr:
        peak_lr = 1e-3
        scheduler = "inv-sqrt"
        warmup_mode = "linear"
        warmup_steps = 4000
        warmup_start_lr = 0.0

    class optim:
        adam_beta2 = 0.98
        adam_eps = 1e-8
        weight_decay = 1e-4
        label_smoothing = 0.1

    class ema:
        decay = 0.9999

    class nn:
        model = "character-queries"
        rnn_kind = "lstm"
        activation = "gelu"

        stem_channels = 256
        hidden_size = 256
        num_layers = 3
        num_classifier_layers = 3
        num_heads = 8
        dropout_rate = 0.2

    class checkpoint:
        style = "continue"
        keep = "best"

    class metric:
        class iou:
            # When to calcualte the IoU, because it's fairly slow.
            # One of: "always" | "validation" | "never"
            when = "validation"
            threshold = 0.75


# Convert the stats in the dictionary to a regular dictionary in order to save it.
def convert_dict_with_stats(d: Dict) -> Dict:
    # Create a new dict, because the input should keep the StatsTracker for further use.
    out = {k: v.get() if isinstance(v, StatsTracker) else v for k, v in d.items()}
    return out


def run_epoch(
    data_loader: DataLoader,
    model: nn.Module,
    optimiser: optim.Optimizer,
    device: torch.device,
    logger: lavd.Logger,
    epoch: int,
    lr_scheduler: Optional[BaseLrScheduler],
    label_smoothing: float = 0.0,
    train: bool = True,
    name: str = "",
    amp_scaler: Optional[amp.GradScaler] = None,
    iou_when: str = DEFAULTS.metric.iou.when,
    iou_threshold: float = DEFAULTS.metric.iou.threshold,
    ema_model: Optional[AveragedModel] = None,
):
    # Disables autograd during validation mode
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()

    sampler = (
        data_loader.sampler
        if isinstance(data_loader.sampler, DistributedSampler)
        else None
    )
    if sampler is not None:
        sampler.set_epoch(epoch)
    num_classes = data_loader.dataset.num_classes  # type: ignore

    unwrapped_model = unwrap_model(model)

    losses = []
    ious = []
    num_correct = 0
    num_total = 0
    pbar = logger.progress_bar(
        name,
        total=len(data_loader.dataset),  # type: ignore
        leave=False,
        dynamic_ncols=True,
    )
    for d in data_loader:  # type: Batch
        features = d.features.to(device)
        embeddables = d.embeddables
        if embeddables is not None:
            embeddables = embeddables.to(device)
        target = d.targets.to(device)
        # The last batch may not be a full batch
        curr_batch_size = d.lengths.size(0)
        if unwrapped_model.predict_clusters:
            assert (
                d.clusters is not None
            ), "Clusters cannot be None for a model with predict_clusters"
            clusters = d.clusters.to(device)
        else:
            clusters = None

        # Automatically run it in mixed precision (FP16) if a scaler is given
        with amp.autocast(enabled=amp_scaler is not None):
            # Dimension: batch_size x max_length x num_classes
            output = model(
                features,
                embeddables=embeddables,
                lengths=d.lengths,
                clusters=clusters,
            )

            no_pad_mask = target != -1
            # Remove the padding for the loss calculation
            # This also flattens the output/target but the batch doesn't need to be
            # split into individual samples.
            output_no_pad = output[no_pad_mask]
            target_no_pad = target[no_pad_mask]

            # If there is only a single class, it becomes a binary classification,
            if num_classes == 1:
                # Needs to be float for binary cross-entropy.
                target_no_pad = target_no_pad.to(torch.float)
                # Get rid of the singular class dimension in the outputs for the loss
                loss = F.binary_cross_entropy_with_logits(
                    output_no_pad.squeeze(-1), target_no_pad
                )
                # The prediction is True when >= 0.5 after applying sigmoid, which can
                # be achieved just as easily by rounding the values.
                preds_no_pad = torch.round(torch.sigmoid(output_no_pad.squeeze(-1)))
            else:
                # Weigh classes inverse proportionally to the number of occurrences.
                # To counteract strong imbalance in favour of non-boundary class.
                class_weights = 1 - torch.bincount(
                    target_no_pad,
                    # Minimum length is needed when not all classes are represented.
                    # This has to be the last class (unknown), but that is guaranteed.
                    minlength=output_no_pad.size(-1),
                ) / torch.numel(target_no_pad)
                loss = F.cross_entropy(
                    output_no_pad,
                    target_no_pad,
                    label_smoothing=label_smoothing,
                    weight=class_weights.to(output_no_pad.dtype),
                )
                _, preds_no_pad = torch.max(output_no_pad, dim=-1)

            with torch.inference_mode():
                if iou_when == "always" or (iou_when == "validation" and not train):
                    prediction_segments = create_segments(
                        output,
                        d.lengths,
                        clusters=clusters,
                    )
                    for pred_seg, seg in zip(prediction_segments, d.segments):
                        ious.append(points_iou(pred_seg, seg, threshold=iou_threshold))
                num_correct += int(torch.sum(preds_no_pad == target_no_pad))
                num_total += torch.numel(target_no_pad)
        losses.append(loss.item())

        if train:
            if torch.isnan(loss) or torch.isinf(loss):
                breakpoint("Loss is NaN")
            if lr_scheduler is not None:
                lr_scheduler.adjust_lr()
            optimiser.zero_grad()
            if amp_scaler is None:
                loss.backward()
                # Clip gradients to avoid exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
            else:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimiser)
                # Clip gradients to avoid exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                amp_scaler.step(optimiser)
                amp_scaler.update()
            if ema_model is not None:
                ema_model.update_parameters(model)

        pbar.update(
            curr_batch_size
            if sampler is None
            else curr_batch_size * sampler.num_replicas
        )
    pbar.close()

    result = dict(
        loss=torch.mean(torch.tensor(losses)).item(),
        accuracy=0 if num_total == 0 else num_correct / num_total,
        iou=torch.mean(torch.tensor(ious)).item() if len(ious) > 0 else None,
    )
    # Gather the metrics onto the primary process
    result = sync_dict_values(result, device=device)
    return result


def train(
    logger: lavd.Logger,
    model: BaseSegmentationModel,
    optimiser: optim.Optimizer,
    train_data_loader: DataLoader,
    validation_data_loaders: List[DataLoader],
    device: torch.device,
    lr_scheduler: BaseLrScheduler,
    num_epochs: int = DEFAULTS.num_epochs,
    checkpoint: Optional[Dict] = None,
    name: str = "",
    amp_scaler: Optional[amp.GradScaler] = None,
    label_smoothing: float = 0.0,
    keep_checkpoint: str = "latest",
    best_metric: str = METRICS[0].key,
    iou_when: str = DEFAULTS.metric.iou.when,
    iou_threshold: float = DEFAULTS.metric.iou.threshold,
    ema_model: Optional[AveragedModel] = None,
):
    if checkpoint is None:
        start_epoch = 0
        train_stats: Dict = dict(lr=[], stats=StatsTracker.with_metrics(METRICS))
        validation_cp: Dict[str, Dict] = {}
        outdated_validations: List[Dict[str, Dict]] = []
        best_checkpoint: Dict = dict(epoch=0)
    else:
        start_epoch = checkpoint["epoch"]
        train_stats = checkpoint["train"]
        validation_cp = checkpoint["validation"]
        outdated_validations = checkpoint["outdated_validation"]
        best_checkpoint = checkpoint["best"]

    validation_results_dict: Dict[str, Dict] = {}
    for val_data_loader in validation_data_loaders:
        val_name = val_data_loader.dataset.name  # type: ignore
        val_stats = (
            validation_cp[val_name]
            if val_name in validation_cp
            else dict(start=start_epoch, stats=StatsTracker.with_metrics(METRICS))
        )
        validation_results_dict[val_name] = val_stats

    # All validations that are no longer used, will be stored in outdated_validation
    # just to have them available.
    outdated_validations.append(
        {k: v for k, v in validation_cp.items() if k not in validation_results_dict}
    )

    for epoch in range(num_epochs):
        actual_epoch = start_epoch + epoch + 1
        epoch_text = "[{current:>{pad}}/{end}] Epoch {epoch}".format(
            current=epoch + 1,
            end=num_epochs,
            epoch=actual_epoch,
            pad=len(str(num_epochs)),
        )
        logger.set_prefix(epoch_text)
        logger.start(epoch_text, prefix=False)
        start_time = time.time()

        logger.start("Train")
        train_result = run_epoch(
            train_data_loader,
            model,
            optimiser,
            device=device,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
            label_smoothing=label_smoothing,
            train=True,
            name="Train",
            logger=logger,
            amp_scaler=amp_scaler,
            iou_when=iou_when,
            iou_threshold=iou_threshold,
            ema_model=ema_model,
        )
        train_stats["lr"].append(lr_scheduler.lr)
        train_stats["stats"].append(train_result)
        logger.end("Train")

        validation_results = []
        for val_data_loader in validation_data_loaders:
            val_name = val_data_loader.dataset.name  # type: ignore
            val_text = "Validation: {}".format(val_name)
            logger.start(val_text)
            validation_result = run_epoch(
                val_data_loader,
                model if ema_model is None else ema_model,
                optimiser,
                device=device,
                epoch=epoch,
                lr_scheduler=lr_scheduler,
                label_smoothing=label_smoothing,
                train=False,
                name=val_text,
                logger=logger,
                amp_scaler=amp_scaler,
                iou_when=iou_when,
                iou_threshold=iou_threshold,
            )
            validation_results.append(dict(name=val_name, stats=validation_result))
            validation_results_dict[val_name]["stats"].append(validation_result)
            logger.end(val_text)

        with logger.spinner("Checkpoint", placement="right"):
            unwrapped_model = unwrap_model(model if ema_model is None else ema_model)
            best_metric_value = average_checkpoint_metric(
                validation_results,
                key=best_metric,
            )
            if (
                best_checkpoint.get(best_metric) is None
                or best_metric_value > best_checkpoint[best_metric]
            ):
                best_checkpoint = {
                    "epoch": actual_epoch,
                    best_metric: best_metric_value,
                }
            stats_checkpoint: Dict = dict(
                step=lr_scheduler.step,
                epoch=actual_epoch,
                train=convert_dict_with_stats(train_stats),
                validation={
                    k: convert_dict_with_stats(v)
                    for k, v in validation_results_dict.items()
                },
                outdated_validation=outdated_validations,
                best=best_checkpoint,
            )
            model_checkpoint = dict(
                kind=unwrapped_model.kind,
                state=unwrapped_model.state_dict(),
                config=unwrapped_model.config(),  # type: ignore
                features_config=(
                    train_data_loader.dataset.features_config()  # type: ignore
                ),
            )
            if keep_checkpoint == "all":
                logger.save_obj(stats_checkpoint, "stats", step=actual_epoch)
                logger.save_obj(model_checkpoint, "model", step=actual_epoch)
            if keep_checkpoint in ["best", "all"]:
                # Only save it as best if the current epoch is the best
                if stats_checkpoint["best"]["epoch"] == actual_epoch:
                    logger.save_obj(stats_checkpoint, "best/stats")
                    logger.save_obj(model_checkpoint, "best/model")
            if keep_checkpoint in ["latest", "best", "all"]:
                logger.save_obj(stats_checkpoint, "latest/stats")
                logger.save_obj(model_checkpoint, "latest/model")

        with logger.spinner("Logging Data", placement="right"):
            log_results(
                logger,
                actual_epoch,
                dict(lr=lr_scheduler.lr, stats=train_result),
                validation_results,
                metrics=METRICS,
            )

        with logger.spinner("Best Checkpoints", placement="right"):
            log_top_checkpoints(logger, validation_results_dict, METRICS)

        time_difference = time.time() - start_time
        epoch_results = [dict(name="Train", stats=train_result)] + validation_results
        log_epoch_stats(
            logger,
            epoch_results,
            METRICS,
            lr_scheduler=lr_scheduler,
            time_elapsed=time_difference,
        )
        # Report when new best checkpoint was saved
        # Here instead of when saving, to get it after the table of the epoch results.
        if stats_checkpoint["best"]["epoch"] == actual_epoch:
            logger.println(
                (
                    "{icon:>{pad}} New best checkpoint: "
                    "Epoch {num:0>4} — {metric_name} = {metric_value:.5f} {icon}"
                ),
                icon="🔔",
                pad=logger.indent_size,
                num=stats_checkpoint["best"]["epoch"],
                metric_name=METRICS_DICT[best_metric].short_name
                or METRICS_DICT[best_metric].name,
                metric_value=stats_checkpoint["best"].get(best_metric, "N/A"),
            )
        logger.end(epoch_text, prefix=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-train",
        dest="gt_train",
        required=True,
        type=str,
        help="Path to the ground truth JSON file used for training",
    )
    parser.add_argument(
        "--gt-validation",
        dest="gt_validation",
        nargs="+",
        metavar="[NAME=]PATH",
        required=True,
        type=str,
        help=(
            "List of ground truth JSON files used for validation "
            "If no name is specified it uses the name of the ground truth file. "
        ),
    )
    parser.add_argument(
        "--chars",
        dest="chars_file",
        type=str,
        help="Path to TSV file with the available characters",
    )
    parser.add_argument(
        "-n",
        "--num-epochs",
        dest="num_epochs",
        default=DEFAULTS.num_epochs,
        type=int,
        help="Number of epochs to train [Default: {}]".format(DEFAULTS.num_epochs),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        default=DEFAULTS.batch_size,
        type=int,
        help="Size of data batches [Default: {}]".format(DEFAULTS.batch_size),
    )
    parser.add_argument(
        "--stem-channels",
        dest="stem_channels",
        default=DEFAULTS.nn.stem_channels,
        type=int,
        help=(
            "Number of output channels of the stem. "
            "If set to 0, no linear projection is used in at the end of the stem "
            "[Default: {}]"
        ).format(DEFAULTS.nn.stem_channels),
    )
    parser.add_argument(
        "--hidden-size",
        dest="hidden_size",
        default=DEFAULTS.nn.hidden_size,
        type=int,
        help="Number of features for each layer [Default: {}]".format(
            DEFAULTS.nn.hidden_size
        ),
    )
    parser.add_argument(
        "--num-layers",
        dest="num_layers",
        default=DEFAULTS.nn.num_layers,
        type=int,
        help="Number of RNN/Transformer layers [Default: {}]".format(
            DEFAULTS.nn.num_layers
        ),
    )
    parser.add_argument(
        "--num-classifier-layers",
        dest="num_classifier_layers",
        default=DEFAULTS.nn.num_classifier_layers,
        type=int,
        help="Number of classifier layers [Default: {}]".format(
            DEFAULTS.nn.num_classifier_layers
        ),
    )
    parser.add_argument(
        "--num-heads",
        dest="num_heads",
        default=DEFAULTS.nn.num_heads,
        type=int,
        help="Number of attention heads [Default: {}]".format(DEFAULTS.nn.num_heads),
    )
    parser.add_argument(
        "--dropout",
        default=DEFAULTS.nn.dropout_rate,
        dest="dropout_rate",
        type=float,
        help="Dropout probability [Default: {}]".format(DEFAULTS.nn.dropout_rate),
    )
    parser.add_argument(
        "--activation",
        dest="activation",
        default=DEFAULTS.nn.activation,
        choices=ACTIVATION_KINDS,
        help="Activation function to use [Default: {}]".format(DEFAULTS.nn.activation),
    )
    parser.add_argument(
        "--rnn-kind",
        dest="rnn_kind",
        default=DEFAULTS.nn.rnn_kind,
        choices=RNN_KINDS,
        help=(
            "Which kind of RNN to use when the RNN model is selected [Default: {}]"
        ).format(DEFAULTS.nn.rnn_kind),
    )
    parser.add_argument(
        "-w",
        "--workers",
        dest="num_workers",
        default=DEFAULTS.num_workers,
        type=int,
        help="Number of workers for loading the data [Default: {}]".format(
            DEFAULTS.num_workers
        ),
    )
    parser.add_argument(
        "-g",
        "--gpus",
        dest="num_gpus",
        default=DEFAULTS.num_gpus,
        type=int,
        help="Number of GPUs to use [Default: {}]".format(DEFAULTS.num_gpus),
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        default=DEFAULTS.lr.peak_lr,
        dest="lr",
        type=float,
        help="Peak learning rate to use [Default: {}]".format(DEFAULTS.lr.peak_lr),
    )
    parser.add_argument(
        "--lr-scheduler",
        dest="lr_scheduler",
        default=DEFAULTS.lr.scheduler,
        choices=LR_SCHEDULER_KINDS,
        help="Learning rate scheduler kind to use [Default: {}]".format(
            DEFAULTS.lr.scheduler
        ),
    )
    parser.add_argument(
        "--lr-warmup",
        dest="lr_warmup",
        default=DEFAULTS.lr.warmup_steps,
        type=int,
        help="Number of linear warmup steps for the learning rate [Default: {}]".format(
            DEFAULTS.lr.warmup_steps
        ),
    )
    parser.add_argument(
        "--lr-warmup-start-lr",
        dest="lr_warmup_start_lr",
        default=DEFAULTS.lr.warmup_start_lr,
        type=float,
        help="Learning rate to start the warmup from [Default: {}]".format(
            DEFAULTS.lr.warmup_start_lr
        ),
    )
    parser.add_argument(
        "--lr-warmup-mode",
        dest="lr_warmup_mode",
        default=DEFAULTS.lr.warmup_mode,
        choices=LR_WARMUP_MODES,
        help="How the warmup is performed [Default: {}]".format(
            DEFAULTS.lr.warmup_mode
        ),
    )
    parser.add_argument(
        "--adam-beta2",
        dest="adam_beta2",
        default=DEFAULTS.optim.adam_beta2,
        type=float,
        help="β₂ for the Adam optimiser [Default: {}]".format(
            DEFAULTS.optim.adam_beta2
        ),
    )
    parser.add_argument(
        "--adam-eps",
        dest="adam_eps",
        default=DEFAULTS.optim.adam_eps,
        type=float,
        help="Epsilon for the Adam optimiser [Default: {}]".format(
            DEFAULTS.optim.adam_eps
        ),
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        default=DEFAULTS.optim.weight_decay,
        type=float,
        help="Weight decay of the optimiser [Default: {}]".format(
            DEFAULTS.optim.weight_decay
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        dest="label_smoothing",
        default=DEFAULTS.optim.label_smoothing,
        type=float,
        help="Label smoothing value [Default: {}]".format(
            DEFAULTS.optim.label_smoothing
        ),
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint",
        help="Path to the checkpoint to be loaded to resume training",
    )
    parser.add_argument(
        "--no-cuda",
        dest="no_cuda",
        action="store_true",
        help="Do not use CUDA even if it's available",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="no_persistent_workers",
        action="store_true",
        help=(
            "Do not persist workers after the epoch ends but reinitialise them at the "
            "start of every epoch. (Slower but uses much less RAM)"
        ),
    )
    parser.add_argument(
        "--name",
        dest="name",
        type=str,
        help="Name of the experiment",
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        default=DEFAULTS.seed,
        type=int,
        help="Seed for random initialisation [Default: {}]".format(DEFAULTS.seed),
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model_kind",
        default=DEFAULTS.nn.model,
        choices=MODEL_KINDS,
        help="Which kind of model to use [Default: {}]".format(DEFAULTS.nn.model),
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        help="Enable mixed precision training (FP16)",
    )
    parser.add_argument(
        "--checkpoint-style",
        dest="checkpoint_style",
        choices=[
            "continue",
            "restart",
            "merge",
            "reset-classifier",
            "reset-embedding",
            "transfer",
        ],
        type=str,
        default=DEFAULTS.checkpoint.style,
        help=(
            "How to integrate the checkpoint into the current model. "
            "Either continue training directly or only load weights but restart fresh. "
            "`restart` requires the exact same model, whereas `merge` ignores weights "
            "that are not present. "
            "`reset-classifier` resets the classification weights. "
            "`reset-embedding` resets the character embedding. "
            "`transfer` allows to use a different model and uses the weights that "
            "overlap "
            "[Default: {}]"
        ).format(DEFAULTS.checkpoint.style),
    )
    parser.add_argument(
        "--keep-checkpoint",
        dest="keep_checkpoint",
        default=DEFAULTS.checkpoint.keep,
        choices=["latest", "best", "all"],
        help=(
            "Which checkpoints to keep: "
            "`latest` only saves the checkpoint of the latest epoch. "
            "`best` saves the best checkpoint in addition to the latest one. "
            "`all` keeps all checkpoints "
            "[Default: {}]"
        ).format(DEFAULTS.checkpoint.keep),
    )
    parser.add_argument(
        "--best-metric",
        dest="best_metric",
        choices=[m.key for m in METRICS],
        help=(
            "Which metric to use to determine the best checkpoint. "
            "If not given, the first available metric is used."
        ),
    )
    parser.add_argument(
        "--iou",
        dest="iou_when",
        choices=["always", "validation", "never"],
        default=DEFAULTS.metric.iou.when,
        help=(
            "When to calculate the Intersection over Union (IoU). It is fairly slow, "
            "not having to calcualte it for the training set speeds it up quite a bit. "
            "[Default: {}]"
        ).format(DEFAULTS.metric.iou.when),
    )
    parser.add_argument(
        "--iou-threshold",
        dest="iou_threshold",
        type=float,
        default=DEFAULTS.metric.iou.threshold,
        help=(
            "Threshold for the Intersection over Union (IoU) matching. [Default: {}]"
        ).format(DEFAULTS.metric.iou.threshold),
    )
    parser.add_argument(
        "-f",
        "--features",
        dest="features",
        nargs="+",
        type=str,
        default=DEFAULTS.features.selection,
        help=(
            "List of features to be used. Supports offsets by adding the :delta suffix "
            "[Default: {}]"
        ).format(" ".join(DEFAULTS.features.selection)),
    )
    parser.add_argument(
        "--features-normalise",
        dest="features_normalise",
        action="store_true",
        help="Normalise the features",
    )
    parser.add_argument(
        "--ctc",
        dest="ctc_spike_mode",
        choices=CTC_SPIKE_MODES,
        default=DEFAULTS.features.ctc_spike_mode,
        help=(
            "How to create the CTC indices from the spikes "
            "`monotonic` means that the CTC is assigned to all following points until "
            "a new spike is found, whereas `single` only assigns it to the point where "
            "the ctc spike occurred. "
            "[Default: {}]"
        ).format(DEFAULTS.features.ctc_spike_mode),
    )
    parser.add_argument(
        "--ema",
        dest="ema_decay",
        type=float,
        # const with nargs=? is essentially a default when the option is specified
        # without an argument (but remains None when it's not supplied).
        const=DEFAULTS.ema.decay,
        nargs="?",
        help=(
            "Activate expontential moving average (EMA) of model weights. "
            "Optionally, specify the decay / momentum / alpha of the EMA model. "
            "The value should be very close to 1, i.e. at least 3-4 9s after "
            "the decimal point. "
            "If the flag is specified without a value, it defaults to {}."
        ).format(DEFAULTS.ema.decay),
    )
    return parser


def main():
    options = build_parser().parse_args()
    # Make sure when iou is chosen as the best metric, that it has to be calcualted at
    # least during validation.
    assert not (
        options.best_metric == "iou" and options.iou_when == "never"
    ), "`--best-metric iou` cannot be used with `--iou never`"
    use_cuda = torch.cuda.is_available() and not options.no_cuda
    if use_cuda:
        # Somehow this fixes an unknown error on Windows.
        torch.cuda.current_device()

    if use_cuda and options.num_gpus > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        # Manullay adjust the batch size and workers to split amongst the processes.
        options.actual_batch_size = options.batch_size // options.num_gpus
        options.actual_num_workers = (
            options.num_workers + options.num_gpus - 1
        ) // options.num_gpus
        mp.spawn(run, nprocs=options.num_gpus, args=(options, True))
    else:
        options.actual_batch_size = options.batch_size
        options.actual_num_workers = options.num_workers
        run(0, options)


def run(gpu_id: int, options: argparse.Namespace, distributed: bool = False):
    if distributed:
        dist.init_process_group(
            backend="nccl",
            rank=gpu_id,
            world_size=options.num_gpus,
            init_method="env://",
        )
        torch.cuda.set_device(gpu_id)
    torch.manual_seed(options.seed)
    use_cuda = torch.cuda.is_available() and not options.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    logger = lavd.Logger(options.name, disabled=gpu_id != 0)
    # Parser needs to be rebuilt, since it can't be serialised and it is needed to even
    # detect the number of GPUs, but it's only used to log it.
    parser = build_parser() if gpu_id == 0 else None

    amp_scaler = amp.GradScaler() if use_cuda and options.fp16 else None
    persistent_workers = (
        not options.no_persistent_workers and options.actual_num_workers > 0
    )

    checkpoint_stats = None
    model_checkpoint = None
    model_kind = options.model_kind
    if options.checkpoint is not None:
        checkpoint_dir = Path(options.checkpoint)
        assert (
            checkpoint_dir.is_dir()
        ), "-c/--checkpoint expects the checkpoint directory"
        checkpoint_stats = torch.load(checkpoint_dir / "stats.pt", map_location="cpu")
        checkpoint_model = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
        model_checkpoint = checkpoint_model["state"]
        if options.checkpoint_style != "transfer":
            model_kind = checkpoint_model["kind"]
            model_config = checkpoint_model["config"]
    predict_clusters = model_kind == "character-queries"

    best_metric = options.best_metric or METRICS[0].key
    if options.iou_when == "never" and best_metric == "iou":
        for m in METRICS:
            # Choose the first metric that is not related to IoU
            if not m.key.startswith("iou"):
                best_metric = m.key
                break

    spinner = logger.spinner("Loading data", placement="right")
    spinner.start()

    feature_selector = FeatureSelector(
        options.features,
        normalise=options.features_normalise,
        chars=options.chars_file,
        ctc_spike_mode=options.ctc_spike_mode,
    )
    collate = Collate(
        predict_clusters=predict_clusters,
        has_embeddables=feature_selector.has_embeddables,
    )
    train_dataset = SegmentationDataset(
        options.gt_train,
        feature_selector=feature_selector,
        predict_clusters=predict_clusters,
        name="Train",
    )
    train_sampler: Optional[DistributedSampler] = (
        DistributedSampler(train_dataset, num_replicas=options.num_gpus, rank=gpu_id)
        if distributed
        else None
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=options.actual_batch_size,
        # Only shuffle when not using a sampler
        shuffle=train_sampler is None,
        num_workers=options.actual_num_workers,
        sampler=train_sampler,
        pin_memory=use_cuda,
        # Keep workers alive after the epoch ends to avoid re-initialising them.
        # NOTE: If RAM becomes an issue, set this to false.
        persistent_workers=persistent_workers,
        collate_fn=collate,
    )

    # To track duplicates
    validation_by_name: Dict[str, List[Path]] = {}
    validation_data_loaders = []
    for val_gt in options.gt_validation:
        name, gt_path = split_named_arg(val_gt)
        validation_dataset = SegmentationDataset(
            gt_path,
            feature_selector=feature_selector,
            predict_clusters=predict_clusters,
            name=name,
        )
        if validation_dataset.name not in validation_by_name:
            validation_by_name[validation_dataset.name] = []
        validation_by_name[validation_dataset.name].append(
            validation_dataset.groundtruth
        )

        validation_sampler: Optional[DistributedSampler] = (
            DistributedSampler(
                validation_dataset, num_replicas=options.num_gpus, rank=gpu_id
            )
            if distributed
            else None
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=options.actual_batch_size,
            # Only shuffle when not using a sampler
            shuffle=validation_sampler is None,
            num_workers=options.actual_num_workers,
            sampler=validation_sampler,
            pin_memory=use_cuda,
            # Keep workers alive after the epoch ends to avoid re-initialising them.
            # NOTE: If RAM becomes an issue, set this to false.
            persistent_workers=persistent_workers,
            collate_fn=collate,
        )
        validation_data_loaders.append(validation_data_loader)
    spinner.stop()

    validation_duplicates = {
        name: paths for name, paths in validation_by_name.items() if len(paths) > 1
    }
    if len(validation_duplicates) > 0:
        raise ValueError(
            "Duplicate validation names were given:\n\n{duplicates}".format(
                duplicates="\n\n".join(
                    [
                        "# {name}\n{paths}".format(
                            name=name, paths="\n".join([f"    - {p}" for p in paths])
                        )
                        for name, paths in validation_duplicates.items()
                    ]
                )
            )
        )

    allow_unused_weights = options.checkpoint_style in [
        "merge",
        "reset-classifier",
        "reset-embedding",
        "transfer",
    ]
    # Arguments that all models share
    model_kwargs = dict(
        num_classes=train_dataset.num_classes,
        in_channels=feature_selector.num_pure_features(),
        num_chars=feature_selector.num_chars(),
        stem_channels=options.stem_channels,
        hidden_size=options.hidden_size,
        num_layers=options.num_layers,
        classifier_channels=options.hidden_size,
        classifier_layers=options.num_classifier_layers,
        activation=options.activation,
        dropout_rate=options.dropout_rate,
        checkpoint=model_checkpoint,
        allow_unused_weights=allow_unused_weights,
        reset_classifier=options.checkpoint_style == "reset-classifier",
        reset_embedding=options.checkpoint_style == "reset-embedding",
    )
    # Model specific arguments
    if model_kind == "rnn":
        model_kwargs.update(
            dict(
                rnn_kind=options.rnn_kind,
            )
        )
    elif model_kind == "character-queries" or model_kind == "transformer":
        model_kwargs.update(
            dict(
                num_heads=options.num_heads,
            )
        )
    model = create_model(model_kind, **model_kwargs)
    model_config = model.config()
    model = model.to(device)

    if checkpoint_stats is not None:
        if options.checkpoint_style == "continue":
            resume_text = "Resuming from - Epoch {epoch}".format(
                epoch=checkpoint_stats["epoch"]
            )
            logger.set_prefix(resume_text)
            checkpoint_stats["train"]["stats"] = StatsTracker.from_dict(
                checkpoint_stats["train"]["stats"], metrics=METRICS
            )
            for val_stats in checkpoint_stats["validation"].values():
                val_stats["stats"] = StatsTracker.from_dict(
                    val_stats["stats"], metrics=METRICS
                )
            epoch_results = [
                dict(name="Train", stats=checkpoint_stats["train"]["stats"].last())
            ] + [
                dict(name=val_name, stats=val_stats["stats"].last())
                for val_name, val_stats in checkpoint_stats["validation"].items()
            ]
            log_epoch_stats(logger, epoch_results, METRICS)
            best_metric_value = checkpoint_stats["best"].get(best_metric)
            logger.println(
                (
                    "{pad}Best checkpoint: "
                    "Epoch {num:0>4} — {metric_name} = {metric_value}"
                ),
                pad=" " * logger.indent_size,
                num=checkpoint_stats["best"]["epoch"],
                metric_name=METRICS_DICT[best_metric].short_name
                or METRICS_DICT[best_metric].name,
                metric_value="N/A"
                if best_metric_value is None
                else f"{best_metric_value:.5f}",
            )
        else:
            logger.println("Starting training with pre-trained weights")
            # Reset the checkpoint, since this is essentially a new training just
            # starting with pre-trained weights.
            checkpoint_stats = None

    optimiser = optim.AdamW(
        model.parameters(),
        lr=options.lr,
        betas=(0.9, options.adam_beta2),
        eps=options.adam_eps,
    )

    if distributed:
        model = DistributedDataParallel(  # type: ignore
            model, device_ids=[gpu_id], find_unused_parameters=False
        )

    initial_step = 0
    if checkpoint_stats is not None:
        checkpoint_step = checkpoint_stats.get("step")
        if checkpoint_step is not None:
            initial_step = checkpoint_step
    lr_scheduler = create_lr_scheduler(
        options.lr_scheduler,
        optimiser,
        peak_lr=options.lr,
        initial_step=initial_step,
        warmup_steps=options.lr_warmup,
        warmup_start_lr=options.lr_warmup_start_lr,
        warmup_mode=options.lr_warmup_mode,
        total_steps=len(train_data_loader) * options.num_epochs,
        end_lr=1e-8,
        # To not crash when choosing schedulers that don't support all arguments.
        allow_extra_args=True,
    )

    ema_model = (
        None
        if options.ema_decay is None
        else AveragedModel(model, ema_alpha=options.ema_decay)
    )

    # Log the details about the experiment
    validation_details = {
        data_loader.dataset.name: {  # type: ignore
            "Size": len(data_loader.dataset),  # type: ignore
        }
        for data_loader in validation_data_loaders
    }
    log_experiment(
        logger,
        model=dict(kind=model_kind, config=model_config),
        train=dict(Size=len(train_dataset)),
        validation=validation_details,
        options=options,
        chars=feature_selector.chars,
        feature_selector=feature_selector,
        lr_scheduler=lr_scheduler,
    )
    logger.log_command(parser, options)

    train(
        logger,
        model,
        optimiser,
        train_data_loader,
        validation_data_loaders,
        lr_scheduler=lr_scheduler,
        label_smoothing=options.label_smoothing,
        device=device,
        num_epochs=options.num_epochs,
        checkpoint=checkpoint_stats,
        name=options.name,
        amp_scaler=amp_scaler,
        keep_checkpoint=options.keep_checkpoint,
        best_metric=best_metric,
        iou_when=options.iou_when,
        iou_threshold=options.iou_threshold,
        ema_model=ema_model,
    )


if __name__ == "__main__":
    main()
