
def to_point_form(left, top, width, height):
    """
    Convert [left, top, width, height] form box to [xmin, ymin, xmax, ymax].
    """

    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1  # noqa

    right = right + margin

    return left, top, right, bottom
