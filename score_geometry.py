import math
from bisect import bisect_left, bisect_right
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image


class Staff(NamedTuple):
    left: int
    right: int
    top: float
    spacing: float


class Notation(NamedTuple):
    angle: float | None
    vertical_angle: float | None
    flip: bool | None
    staff_lines: int


def analyse_notation(image, initial_angle):
    """Identify the page orientation, then measure corrections on the upright score."""
    gray = image.convert("L")
    gray.thumbnail((2400, 2400), Image.Resampling.BOX)
    seed = initial_angle if initial_angle is not None else 0
    level, ink, horizontal, staffs = staff_geometry(gray, seed)
    if not staffs:
        return Notation(initial_angle, None, None, 0)

    flip = detect_flip(ink, staffs, 0)
    if flip is None:
        # A strong vertical lean can obscure the clef shape. Compensate in the
        # orientation check before measuring the corrections to apply.
        lean_angle = measure_barline_angle(level, staffs)
        if lean_angle:
            flip = detect_flip(ink, staffs, math.tan(math.radians(lean_angle)))
    if flip:
        upright = gray.transpose(Image.Transpose.ROTATE_180)
        level, _, horizontal, staffs = staff_geometry(upright, seed)

    slopes = staff_line_angles(horizontal, staffs)
    angle = round(float(np.clip(seed + np.median(slopes), -15, 15)), 3) if slopes else initial_angle
    barline_angle = measure_barline_angle(level, staffs) if staffs and angle is not None else None
    vertical_angle = None
    if barline_angle is not None:
        # The barlines were measured using the coarse rotation. Express their lean
        # relative to the refined horizontal rotation that will actually be applied.
        corrected_reference = round(float(barline_angle + angle - seed), 3)
        if abs(corrected_reference) <= 15:
            vertical_angle = corrected_reference
    return Notation(angle, vertical_angle, flip, len(slopes))


def staff_geometry(gray, seed):
    width, height = gray.size
    rotation = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), seed, 1)
    level = cv2.warpAffine(np.asarray(gray), rotation, gray.size, borderValue=255)
    ink = cv2.adaptiveThreshold(level, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 13)
    # A one-pixel vertical tolerance keeps faint, slightly curved lines connected.
    line_ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(14, width // 8), 1))
    horizontal = cv2.morphologyEx(line_ink, cv2.MORPH_OPEN, kernel)
    return level, ink, horizontal, find_staffs(horizontal)


def find_staffs(horizontal):
    height, width = horizontal.shape
    _, _, stats, centers = cv2.connectedComponentsWithStats(horizontal)
    lines = sorted((float(row), int(x), int(x + w))
                   for (x, _, w, h, _), (_, row) in zip(stats[1:], centers[1:])
                   if w >= width * 0.15 and h < max(4, width * 0.009)
                   and height * 0.035 < row < height * 0.965)
    rows = [row for row, _, _ in lines]
    staffs = []
    for first, (top, left, right) in enumerate(lines):
        if any(abs(top - staff.top) < 3.6 * staff.spacing
               and left < staff.right and right > staff.left for staff in staffs):
            continue
        overlap = (right - left) * 0.6
        max_spacing = min(48, height * 0.045)
        for second in range(bisect_left(rows, top + 3, first + 1),
                            bisect_right(rows, top + max_spacing, first + 1)):
            spacing = rows[second] - top
            tolerance = max(1.3, spacing * 0.19)
            group = [lines[first], lines[second]]
            for number in range(2, 5):
                target = top + number * spacing
                start = bisect_left(rows, target - tolerance, second + 1)
                end = bisect_right(rows, target + tolerance, start)
                candidates = [line for line in lines[start:end]
                              if min(right, line[2]) - max(left, line[1]) >= overlap]
                if not candidates:
                    break
                group.append(min(candidates, key=lambda line: abs(line[0] - target)))
            if len(group) != 5:
                continue
            line_rows, starts, ends = zip(*group)
            if min(ends) - max(starts) >= overlap:
                staffs.append(Staff(int(np.median(starts)), int(np.median(ends)), top,
                                    (line_rows[-1] - top) / 4))
                break
    return staffs


def staff_line_angles(horizontal, staffs):
    angles = []
    for left, right, top, spacing in staffs:
        for number in range(5):
            center = top + number * spacing
            radius = max(1.5, spacing * 0.23)
            low, high = max(0, math.floor(center - radius)), math.ceil(center + radius) + 1
            y, x = np.nonzero(horizontal[low:high, left:right])
            if len(x) < 2 or np.ptp(x) < (right - left) * 0.6:
                continue
            points = np.column_stack((x, y)).astype(np.float32)
            vx, vy, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            angles.append(math.degrees(math.atan2(vy, vx)))
    return angles


def measure_barline_angle(level, staffs):
    dx = cv2.Sobel(level, cv2.CV_16S, 1, 0)
    dy = cv2.Sobel(level, cv2.CV_16S, 0, 1)
    edges = cv2.Canny(level, 45, 125)
    edges[np.abs(dx) <= np.abs(dy)] = 0
    spacing = float(np.median([staff.spacing for staff in staffs]))
    segments = cv2.HoughLinesP(edges, 1, np.pi / 720, max(12, round(2.5 * spacing)),
                               minLineLength=max(12, round(3.4 * spacing)),
                               maxLineGap=max(3, round(0.65 * spacing)))
    if segments is None:
        return None

    candidates = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4).tolist():
        if y1 > y2:
            x1, x2, y1, y2 = x2, x1, y2, y1
        span = y2 - y1
        if not span or abs(x2 - x1) > 0.30 * span:
            continue
        slope = (x2 - x1) / span
        # A barline crosses nearly the full staff. Exclude the outer edges where
        # binding marks and borders can look like measure bars.
        if not any(y1 <= top + 0.4 * space and y2 >= top + 3.6 * space
                   and left + 2 * space < x1 + (top + 2 * space - y1) * slope < right - 2 * space
                   for left, right, top, space in staffs):
            continue
        refined = refine_barline(dx, x1, y1, y2, slope)
        if refined is not None:
            angle, column = refined
            candidates.append((angle, min(span, 300), column))

    if len(candidates) < 3:
        return None
    angles, weights, columns = np.array(sorted(candidates)).T
    cumulative = np.cumsum(weights)
    median = angles[np.searchsorted(cumulative, cumulative[-1] / 2)]
    agreement = np.abs(angles - median) <= 0.75
    if weights[agreement].sum() < cumulative[-1] * 0.65 or np.ptp(columns[agreement]) < spacing:
        return None
    return float(median)


def refine_barline(gradient, x, first_row, last_row, slope):
    rows = np.arange(first_row, last_row + 1)
    centers = np.rint(x + (rows - first_row) * slope).astype(int)
    columns = np.clip(centers[:, None] + np.arange(-2, 3), 0, gradient.shape[1] - 1)
    samples = gradient[rows[:, None], columns]
    # Follow one edge of the same stroke with subpixel centers of the gradient.
    sign = 1 if samples.sum() > 0 else -1
    weights = np.maximum(sign * samples - 40, 0)
    totals = weights.sum(axis=1)
    valid = totals > 80
    if np.count_nonzero(valid) < len(rows) * 0.65:
        return None
    positions = (weights[valid] * columns[valid]).sum(axis=1) / totals[valid]
    points = np.column_stack((rows[valid], positions)).astype(np.float32)
    vy, vx, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    angle = math.degrees(math.atan2(vx, vy))
    if abs(angle) > 15:
        return None
    column = np.mean(positions - (rows[valid] - gradient.shape[0] / 2) * (vx / vy))
    return angle, float(column)


# The three cells contain G, F and C clefs rendered from Bravura at this scale.
STAFF_SPACE = 24


@lru_cache(maxsize=1)
def clef_templates():
    path = Path(__file__).with_name("score-assets") / "clefs.png"
    with Image.open(path) as sheet:
        cell_width = 3 * STAFF_SPACE
        templates = []
        for x in range(0, sheet.width, cell_width):
            cell = sheet.crop((x, 0, x + cell_width, sheet.height))
            bounds = cell.getbbox()
            template = np.asarray(cell.crop(bounds))
            mask = np.full(template.shape, 255, np.uint8)
            for staff_row in range(3, 8):
                row = staff_row * STAFF_SPACE - bounds[1]
                # Compare clef shapes without the horizontal staff lines.
                mask[max(0, row - 3):max(0, row + 4)] = 0
            templates.append((bounds[1], template, mask))
        return templates


def clef_score(ink, staff, lean):
    left, _, top, spacing = staff
    scale = spacing / STAFF_SPACE
    # Sample a straightened clef around the midpoint of the tilted staff edge.
    transform = np.float32([[scale, scale * lean, left - (1 + 5 * lean) * spacing],
                            [0, scale, top - 3 * spacing]])
    strip = cv2.warpAffine(ink, transform, (5 * STAFF_SPACE, 10 * STAFF_SPACE),
                           flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    best = 0.0
    for y, template, mask in clef_templates():
        region = strip[y - 6:y + template.shape[0] + 7]
        scores = cv2.matchTemplate(region, template, cv2.TM_CCORR_NORMED, mask=mask)
        best = max(best, float(np.max(scores[np.isfinite(scores)], initial=0)))
    return best


def detect_flip(ink, staffs, lean):
    height, width = ink.shape
    turned = cv2.rotate(ink, cv2.ROTATE_180)
    upright, upside_down = [], []
    for left, right, top, spacing in staffs:
        normal = clef_score(ink, Staff(left, right, top, spacing), lean)
        rotated = clef_score(turned, Staff(width - right, width - left,
                                          height - 1 - top - 4 * spacing, spacing), lean)
        if normal >= 0.75 and normal - rotated >= 0.12:
            upright.append(normal)
        elif rotated >= 0.75 and rotated - normal >= 0.12:
            upside_down.append(rotated)
    for flip, votes, other in ((False, upright, upside_down), (True, upside_down, upright)):
        if votes and len(votes) > 2 * len(other) and (len(votes) >= 2 or votes[0] >= 0.84):
            return flip
    return None
