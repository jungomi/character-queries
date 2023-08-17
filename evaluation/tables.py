import re
from typing import List, Union

table_separator_regex = re.compile("[^|]")


def round_float(num: float, places: int = 2) -> float:
    return round(num * 10**places) / 10**places


def create_markdown_table(
    header: List[str], rows: List[List[Union[str, float, int]]], precision: int = 5
) -> List[str]:
    output = []
    rows_formatted = []
    # Keeps track of the longest field length in each column, so that the table can
    # be aligned nicely.
    column_widths = [len(name) for name in header]
    for row in rows:
        row_formatted = []
        for column, field in enumerate(row):
            if isinstance(field, str):
                field_str = field
            elif isinstance(field, int):
                field_str = str(field)
            elif isinstance(field, float):
                field_str = "{num:.{precision}f}".format(num=field, precision=precision)
            row_formatted.append(field_str)
            # Check whether the field is longer than current maximum column width
            # and expand it if necessary.
            if len(field_str) > column_widths[column]:
                column_widths[column] = len(field_str)
        rows_formatted.append(row_formatted)
    header = pad_table_row(header, column_widths)
    header_line = "| {fields} |".format(fields=" | ".join(header))
    output.append(header_line)
    separator = table_separator_regex.sub("-", header_line)
    output.append(separator)
    for r in rows_formatted:
        r = pad_table_row(r, column_widths)
        line = "| {fields} |".format(fields=" | ".join(r))
        output.append(line)
    return output


def pad_table_row(row: List[str], widths: List[int], value: str = " ") -> List[str]:
    return [
        "{field:{pad}<{width}}".format(field=field, pad=value, width=width)
        for field, width in zip(row, widths)
    ]
