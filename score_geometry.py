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


def analyse_notation(image, initial_angle):
    """Measure the staff angle and use clefs to identify a 180° rotation."""
    gray = image.convert("L")
    gray.thumbnail((2400, 2400), Image.Resampling.BOX)
    width, height = gray.size
    angle = initial_angle if initial_angle is not None else 0
    rotation = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1)
    level = cv2.warpAffine(np.asarray(gray), rotation, gray.size, borderValue=255)
    ink = cv2.adaptiveThreshold(level, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 13)
    # A one-pixel vertical tolerance keeps faint, slightly curved lines connected.
    line_ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(14, width // 8), 1))
    horizontal = cv2.morphologyEx(line_ink, cv2.MORPH_OPEN, kernel)
    staffs = find_staffs(horizontal)
    if not staffs:
        return initial_angle, None, 0

    slopes = staff_line_angles(horizontal, staffs)
    if slopes:
        angle = round(float(np.clip(angle + np.median(slopes), -15, 15)), 3)
    else:
        angle = initial_angle
    return angle, detect_flip(ink, staffs), len(slopes)


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


def clef_score(ink, staff):
    left, _, top, spacing = staff
    scale = spacing / STAFF_SPACE
    transform = np.float32([[scale, 0, left - spacing], [0, scale, top - 3 * spacing]])
    strip = cv2.warpAffine(ink, transform, (5 * STAFF_SPACE, 10 * STAFF_SPACE),
                           flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    best = 0.0
    for y, template, mask in clef_templates():
        region = strip[y - 6:y + template.shape[0] + 7]
        scores = cv2.matchTemplate(region, template, cv2.TM_CCORR_NORMED, mask=mask)
        best = max(best, float(np.max(scores[np.isfinite(scores)], initial=0)))
    return best


def detect_flip(ink, staffs):
    height, width = ink.shape
    turned = cv2.rotate(ink, cv2.ROTATE_180)
    upright, upside_down = [], []
    for left, right, top, spacing in staffs:
        normal = clef_score(ink, Staff(left, right, top, spacing))
        rotated = clef_score(turned, Staff(width - right, width - left,
                                          height - 1 - top - 4 * spacing, spacing))
        if normal >= 0.75 and normal - rotated >= 0.12:
            upright.append(normal)
        elif rotated >= 0.75 and rotated - normal >= 0.12:
            upside_down.append(rotated)
    for flip, votes, other in ((False, upright, upside_down), (True, upside_down, upright)):
        if votes and len(votes) > 2 * len(other) and (len(votes) >= 2 or votes[0] >= 0.84):
            return flip
    return None
