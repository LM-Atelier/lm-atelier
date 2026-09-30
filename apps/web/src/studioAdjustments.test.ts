/** The preview's arithmetic, checked against the pixels the server makes. */

import { describe, expect, it } from "vitest";
import {
  adjustPixels, channelTables, grainLevel, grainOffsets, isNeutral, levelEnds, NEUTRAL_ADJUSTMENTS,
} from "./studioAdjustments";
import type { StudioColorAdjustments } from "./types";

// The same pixels and results test_studio_adjustments.py checks the server
// against. If either side's arithmetic moves, one of the two suites fails. The
// last three pixels sit where rounding the saturation mix to single precision
// changes the answer, so they fail if it is ever done in double.
const PIXELS = [
  [200, 100, 50], [10, 240, 128], [128, 128, 128], [255, 0, 255], [0, 0, 0],
  [255, 255, 255], [37, 91, 203], [0, 170, 0], [210, 253, 126], [169, 45, 156],
];
const CASES: Array<[string, Partial<StudioColorAdjustments>, number[][]]> = [
  ["brighter", { brightness: 40 }, [
    [255, 132, 66], [13, 255, 169], [169, 169, 169], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [49, 120, 255], [0, 224, 0], [255, 255, 166], [223, 59, 206]]],
  ["flatter", { contrast: -30 }, [
    [186, 105, 65], [32, 219, 128], [128, 128, 128], [231, 24, 231], [24, 24, 24],
    [231, 231, 231], [54, 98, 189], [24, 162, 24], [195, 229, 126], [161, 60, 151]]],
  ["grey", { saturation: -100 }, [
    [124, 124, 124], [158, 158, 158], [128, 128, 128], [105, 105, 105], [0, 0, 0],
    [255, 255, 255], [88, 88, 88], [100, 100, 100], [226, 226, 226], [95, 95, 95]]],
  ["richer", { saturation: 60 }, [
    [245, 85, 5], [0, 255, 110], [128, 128, 128], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [6, 92, 255], [0, 212, 0], [200, 255, 66], [213, 15, 192]]],
  ["paler", { saturation: -41 }, [
    [168, 109, 80], [70, 206, 140], [128, 128, 128], [193, 43, 193], [0, 0, 0],
    [255, 255, 255], [57, 89, 155], [41, 141, 41], [216, 241, 167], [138, 65, 130]]],
  ["livelier", { saturation: 8 }, [
    [206, 98, 44], [0, 246, 125], [128, 128, 128], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [32, 91, 212], [0, 175, 0], [208, 255, 117], [174, 40, 160]]],
  ["warmer", { warmth: 50 }, [
    [215, 99, 46], [11, 237, 118], [138, 126, 118], [255, 0, 235], [0, 0, 0],
    [255, 251, 235], [40, 90, 187], [0, 168, 0], [226, 249, 116], [182, 44, 144]]],
  ["magenta", { tint: 60 }, [
    [227, 95, 57], [11, 227, 145], [145, 121, 145], [255, 0, 255], [0, 0, 0],
    [255, 241, 255], [42, 86, 230], [0, 161, 0], [238, 240, 143], [192, 43, 177]]],
  ["greener", { tint: -100 }, [
    [160, 108, 40], [8, 255, 102], [102, 138, 102], [204, 0, 204], [0, 0, 0],
    [204, 255, 204], [30, 98, 162], [0, 184, 0], [168, 255, 101], [135, 49, 125]]],
  ["tinted and warmed", { tint: 35, warmth: -45, brightness: 15 }, [
    [232, 107, 66], [12, 255, 169], [148, 137, 169], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [43, 98, 255], [0, 183, 0], [243, 255, 167], [196, 48, 207]]],
  ["everything", { brightness: -20, contrast: 35, saturation: 25, warmth: -40 }, [
    [200, 69, 7], [0, 251, 114], [102, 107, 124], [255, 0, 255], [0, 0, 0],
    [238, 247, 255], [0, 66, 250], [0, 168, 0], [186, 252, 95], [166, 1, 177]]],
  ["lifted shadows", { shadows: 100 }, [
    [209, 137, 82], [19, 241, 160], [160, 160, 160], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [64, 129, 211], [0, 189, 0], [217, 253, 158], [188, 76, 180]]],
  ["deeper shadows", { shadows: -60 }, [
    [194, 78, 31], [4, 240, 109], [109, 109, 109], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [21, 68, 198], [0, 159, 0], [206, 253, 107], [157, 27, 142]]],
  ["recovered highlights", { highlights: -100 }, [
    [166, 76, 42], [10, 227, 96], [96, 96, 96], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [32, 70, 170], [0, 132, 0], [179, 251, 95], [131, 38, 119]]],
  ["brighter highlights", { highlights: 45 }, [
    [215, 111, 54], [10, 246, 142], [142, 142, 142], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [39, 100, 218], [0, 187, 0], [224, 254, 140], [186, 48, 173]]],
  ["vivid", { vibrance: 60 }, [
    [219, 94, 31], [9, 241, 126], [128, 128, 128], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [26, 91, 221], [0, 184, 0], [205, 254, 96], [192, 30, 174]]],
  ["muted", { vibrance: -60 }, [
    [181, 106, 68], [19, 235, 130], [128, 128, 128], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [47, 90, 179], [20, 156, 20], [215, 244, 156], [146, 60, 137]]],
  ["vivid and paler", { vibrance: 80, saturation: -30 }, [
    [202, 99, 47], [34, 230, 131], [128, 128, 128], [223, 22, 223], [0, 0, 0],
    [255, 255, 255], [36, 91, 203], [14, 170, 14], [208, 251, 119], [173, 41, 160]]],
  ["toned and graded", { shadows: 50, highlights: -40, contrast: 20, saturation: -30, warmth: 25 }, [
    [184, 112, 74], [48, 222, 136], [135, 131, 127], [210, 31, 210], [0, 0, 0],
    [255, 255, 255], [54, 95, 164], [29, 148, 29], [223, 247, 156], [151, 62, 134]]],
  ["faded", { blacks: 60, whites: -30 }, [
    [193, 116, 77], [46, 224, 137], [137, 137, 137], [236, 38, 236], [38, 38, 38],
    [236, 236, 236], [67, 109, 196], [38, 170, 38], [201, 234, 136], [169, 73, 159]]],
  ["deeper blacks and brighter whites", { blacks: -50, whites: 70 }, [
    [240, 97, 26], [0, 255, 137], [137, 137, 137], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [7, 84, 244], [0, 197, 0], [254, 255, 134], [196, 19, 177]]],
  ["brighter with white held down", { brightness: 40, whites: -80 }, [
    [204, 106, 53], [11, 204, 135], [135, 135, 135], [204, 0, 204], [0, 0, 0],
    [204, 204, 204], [39, 96, 204], [0, 179, 0], [204, 204, 133], [178, 48, 165]]],
];

// A picture with edges in it, for sharpness, which reads each pixel's
// neighbors: the same pixels and results test_studio_adjustments.py checks the
// server against, opaque and as a cutout. In the cutout some pixels are partly
// transparent, and some wholly so with a color hidden under them.
const SHARP_WIDTH = 6;
const SHARP_COLORS = [
  [12, 40, 200], [60, 60, 60], [200, 30, 30], [250, 250, 250], [0, 0, 0], [90, 180, 40],
  [30, 30, 30], [220, 200, 10], [128, 128, 128], [15, 90, 210], [240, 120, 60], [33, 66, 99],
  [255, 255, 255], [10, 10, 10], [180, 40, 160], [70, 200, 120], [0, 255, 0], [128, 0, 255],
  [45, 45, 200], [150, 150, 20], [5, 5, 5], [250, 10, 120], [100, 100, 100], [200, 200, 200],
  [77, 22, 11], [0, 128, 255], [255, 128, 0], [60, 30, 90], [210, 210, 40], [18, 180, 180],
];
const SHARP_ALPHAS = [
  255, 255, 255, 255, 0, 255,
  255, 128, 255, 64, 255, 255,
  0, 255, 200, 255, 17, 255,
  255, 255, 255, 255, 128, 0,
  255, 255, 0, 255, 255, 255,
];
const SHARP_CASES: Array<[string, Partial<StudioColorAdjustments>, boolean, number[][]]> = [
  ["crisper", { sharpness: 60 }, false, [
    [12, 40, 200, 255], [60, 60, 60, 255], [200, 30, 30, 255], [250, 250, 250, 255], [0, 0, 0, 255], [90, 180, 40, 255],
    [30, 30, 30, 255], [255, 255, 0, 255], [125, 138, 138, 255], [0, 66, 255, 255], [255, 119, 39, 255], [33, 66, 99, 255],
    [255, 255, 255, 255], [0, 0, 0, 255], [221, 15, 199, 255], [50, 247, 126, 255], [0, 255, 0, 255], [128, 0, 255, 255],
    [45, 45, 200, 255], [184, 186, 0, 255], [0, 0, 0, 255], [255, 0, 142, 255], [85, 79, 94, 255], [200, 200, 200, 255],
    [77, 22, 11, 255], [0, 128, 255, 255], [255, 128, 0, 255], [60, 30, 90, 255], [210, 210, 40, 255], [18, 180, 180, 255]]],
  ["softer", { sharpness: -45 }, false, [
    [12, 40, 200, 255], [60, 60, 60, 255], [200, 30, 30, 255], [250, 250, 250, 255], [0, 0, 0, 255], [90, 180, 40, 255],
    [30, 30, 30, 255], [176, 155, 37, 255], [130, 119, 120, 255], [59, 108, 175, 255], [177, 120, 75, 255], [33, 66, 99, 255],
    [255, 255, 255, 255], [57, 48, 41, 255], [148, 58, 130, 255], [84, 164, 115, 255], [44, 202, 47, 255], [128, 0, 255, 255],
    [45, 45, 200, 255], [124, 122, 51, 255], [54, 32, 33, 255], [196, 43, 102, 255], [111, 115, 104, 255], [200, 200, 200, 255],
    [77, 22, 11, 255], [0, 128, 255, 255], [255, 128, 0, 255], [60, 30, 90, 255], [210, 210, 40, 255], [18, 180, 180, 255]]],
  ["crisper, warmer and paler", { sharpness: 100, warmth: 30, saturation: -20 }, false, [
    [20, 41, 162, 255], [62, 60, 57, 255], [183, 40, 39, 255], [253, 248, 241, 255], [0, 0, 0, 255], [102, 170, 58, 255],
    [30, 30, 29, 255], [255, 255, 15, 255], [130, 144, 140, 255], [0, 46, 225, 255], [255, 132, 55, 255], [39, 63, 87, 255],
    [254, 253, 245, 255], [0, 0, 0, 255], [227, 20, 191, 255], [68, 255, 138, 255], [0, 255, 0, 255], [120, 13, 208, 255],
    [50, 48, 165, 255], [210, 204, 0, 255], [0, 0, 0, 255], [255, 0, 140, 255], [77, 65, 82, 255], [206, 198, 192, 255],
    [71, 25, 16, 255], [20, 122, 215, 255], [234, 131, 30, 255], [59, 33, 78, 255], [213, 204, 68, 255], [41, 169, 163, 255]]],
  ["a crisper cutout", { sharpness: 60 }, true, [
    [12, 40, 200, 255], [60, 60, 60, 255], [200, 30, 30, 255], [250, 250, 250, 255], [0, 0, 0, 0], [90, 180, 40, 255],
    [30, 30, 30, 255], [255, 255, 0, 128], [122, 140, 141, 255], [0, 57, 253, 64], [255, 117, 27, 255], [33, 66, 99, 255],
    [255, 255, 255, 0], [0, 0, 0, 255], [223, 17, 203, 200], [36, 255, 125, 255], [0, 255, 0, 17], [128, 0, 255, 255],
    [45, 45, 200, 255], [199, 195, 0, 255], [0, 0, 0, 255], [255, 0, 138, 255], [78, 95, 91, 128], [255, 255, 255, 0],
    [77, 22, 11, 255], [0, 128, 255, 255], [255, 204, 0, 0], [60, 30, 90, 255], [210, 210, 40, 255], [18, 180, 180, 255]]],
  ["the softest cutout", { sharpness: -100 }, true, [
    [12, 40, 200, 255], [60, 60, 60, 255], [200, 30, 30, 255], [250, 250, 250, 255], [0, 0, 0, 0], [90, 180, 40, 255],
    [30, 30, 30, 255], [97, 75, 65, 128], [138, 107, 105, 255], [159, 144, 137, 64], [147, 124, 114, 255], [33, 66, 99, 255],
    [0, 0, 0, 0], [84, 65, 56, 255], [108, 78, 88, 200], [126, 107, 111, 255], [132, 95, 130, 17], [128, 0, 255, 255],
    [45, 45, 200, 255], [68, 74, 83, 255], [92, 56, 75, 255], [134, 67, 90, 255], [136, 107, 114, 128], [0, 0, 0, 0],
    [77, 22, 11, 255], [0, 128, 255, 255], [0, 0, 0, 0], [60, 30, 90, 255], [210, 210, 40, 255], [18, 180, 180, 255]]],
];

// The same picture again, for the vignette, which reads where each pixel is:
// the pixels test_studio_adjustments.py checks the server against.
const VIGNETTE_CASES: Array<[string, Partial<StudioColorAdjustments>, boolean, number[][]]> = [
  ["darker edges", { vignette: 60 }, false, [
    [7, 25, 122, 255], [49, 49, 49, 255], [182, 27, 27, 255], [227, 227, 227, 255], [0, 0, 0, 255], [55, 110, 25, 255],
    [25, 25, 25, 255], [217, 197, 10, 255], [128, 128, 128, 255], [15, 90, 210, 255], [236, 118, 59, 255], [27, 55, 82, 255],
    [229, 229, 229, 255], [10, 10, 10, 255], [180, 40, 160, 255], [70, 200, 120, 255], [0, 255, 0, 255], [115, 0, 229, 255],
    [37, 37, 166, 255], [148, 148, 20, 255], [5, 5, 5, 255], [250, 10, 120, 255], [98, 98, 98, 255], [166, 166, 166, 255],
    [47, 13, 7, 255], [0, 104, 207, 255], [232, 117, 0, 255], [55, 27, 82, 255], [170, 170, 33, 255], [11, 110, 110, 255]]],
  ["lighter edges", { vignette: -45 }, false, [
    [83, 103, 216, 255], [87, 87, 87, 255], [204, 45, 45, 255], [250, 250, 250, 255], [36, 36, 36, 255], [138, 202, 103, 255],
    [59, 59, 59, 255], [220, 201, 13, 255], [128, 128, 128, 255], [15, 90, 210, 255], [240, 122, 62, 255], [62, 90, 119, 255],
    [255, 255, 255, 255], [10, 10, 10, 255], [180, 40, 160, 255], [70, 200, 120, 255], [0, 255, 0, 255], [138, 19, 255, 255],
    [72, 72, 207, 255], [151, 151, 23, 255], [5, 5, 5, 255], [250, 10, 120, 255], [102, 102, 102, 255], [207, 207, 207, 255],
    [129, 90, 82, 255], [36, 146, 255, 255], [255, 137, 17, 255], [73, 45, 101, 255], [216, 216, 70, 255], [87, 202, 202, 255]]],
  ["a cutout with darker edges", { vignette: 100 }, true, [
    [4, 14, 70, 255], [41, 41, 41, 255], [170, 25, 25, 255], [212, 212, 212, 255], [0, 0, 0, 0], [32, 63, 14, 255],
    [21, 21, 21, 255], [214, 195, 10, 128], [128, 128, 128, 255], [15, 90, 210, 64], [234, 117, 58, 255], [24, 47, 71, 255],
    [212, 212, 212, 0], [10, 10, 10, 255], [180, 40, 160, 200], [70, 200, 120, 255], [0, 255, 0, 17], [106, 0, 212, 255],
    [32, 32, 143, 255], [146, 146, 19, 255], [5, 5, 5, 255], [250, 10, 120, 255], [97, 97, 97, 128], [143, 143, 143, 0],
    [27, 7, 4, 255], [0, 88, 175, 255], [217, 109, 0, 0], [51, 25, 76, 255], [144, 144, 27, 255], [7, 63, 63, 255]]],
  ["crisper, livelier and darker at the edges", { sharpness: 40, vibrance: 50, vignette: 30 }, false, [
    [7, 32, 173, 255], [55, 55, 55, 255], [208, 20, 20, 255], [239, 239, 239, 255], [0, 0, 0, 255], [64, 152, 18, 255],
    [28, 28, 28, 255], [253, 239, 0, 255], [124, 136, 134, 255], [0, 75, 255, 255], [253, 113, 28, 255], [21, 62, 103, 255],
    [242, 242, 242, 255], [0, 0, 0, 255], [233, 5, 204, 255], [30, 247, 112, 255], [0, 255, 0, 255], [121, 0, 242, 255],
    [37, 37, 203, 255], [175, 178, 0, 255], [0, 0, 0, 255], [255, 0, 136, 255], [89, 84, 95, 255], [183, 183, 183, 255],
    [74, 13, 2, 255], [0, 116, 231, 255], [244, 122, 0, 255], [62, 23, 102, 255], [193, 193, 25, 255], [9, 152, 152, 255]]],
];

// The same picture again, for grain, which reads where each pixel is too:
// the pixels test_studio_adjustments.py checks the server against.
const GRAIN_CASES: Array<[string, Partial<StudioColorAdjustments>, boolean, number[][]]> = [
  ["grain", { grain: 60 }, false, [
    [2, 30, 190, 255], [61, 61, 61, 255], [203, 33, 33, 255], [255, 255, 255, 255], [16, 16, 16, 255], [91, 181, 41, 255],
    [40, 40, 40, 255], [218, 198, 8, 255], [123, 123, 123, 255], [3, 78, 198, 255], [238, 118, 58, 255], [42, 75, 108, 255],
    [255, 255, 255, 255], [21, 21, 21, 255], [176, 36, 156, 255], [72, 202, 122, 255], [11, 255, 11, 255], [120, 0, 247, 255],
    [53, 53, 208, 255], [152, 152, 22, 255], [7, 7, 7, 255], [255, 24, 134, 255], [88, 88, 88, 255], [199, 199, 199, 255],
    [69, 14, 3, 255], [12, 140, 255, 255], [255, 136, 8, 255], [57, 27, 87, 255], [211, 211, 41, 255], [23, 185, 185, 255]]],
  ["a cutout with the most grain", { grain: 100 }, true, [
    [0, 23, 183, 255], [61, 61, 61, 255], [206, 36, 36, 255], [255, 255, 255, 255], [26, 26, 26, 0], [91, 181, 41, 255],
    [47, 47, 47, 255], [216, 196, 6, 128], [120, 120, 120, 255], [0, 70, 190, 64], [236, 116, 56, 255], [48, 81, 114, 255],
    [255, 255, 255, 0], [29, 29, 29, 255], [173, 33, 153, 200], [73, 203, 123, 255], [19, 255, 19, 17], [114, 0, 241, 255],
    [59, 59, 214, 255], [154, 154, 24, 255], [9, 9, 9, 255], [255, 33, 143, 255], [81, 81, 81, 128], [198, 198, 198, 0],
    [63, 8, 0, 255], [21, 149, 255, 255], [255, 142, 14, 0], [55, 25, 85, 255], [212, 212, 42, 255], [27, 189, 189, 255]]],
  ["a film look", { contrast: 10, saturation: -15, warmth: 10, grain: 35 }, false, [
    [4, 29, 172, 255], [55, 55, 54, 255], [190, 33, 33, 255], [255, 255, 255, 255], [9, 9, 9, 255], [96, 176, 48, 255],
    [29, 29, 29, 255], [222, 201, 28, 255], [126, 125, 123, 255], [10, 78, 185, 255], [235, 122, 67, 255], [36, 65, 94, 255],
    [255, 255, 255, 255], [9, 9, 8, 255], [170, 41, 148, 255], [81, 198, 124, 255], [29, 246, 29, 255], [115, 5, 221, 255],
    [47, 46, 185, 255], [152, 149, 31, 255], [1, 1, 1, 255], [238, 23, 121, 255], [91, 91, 90, 255], [206, 204, 201, 255],
    [62, 11, 1, 255], [22, 131, 239, 255], [244, 136, 27, 255], [51, 23, 77, 255], [216, 213, 58, 255], [31, 178, 176, 255]]],
];

function rgba(pixels: number[][], alpha = 255): Uint8ClampedArray {
  return new Uint8ClampedArray(pixels.flatMap(([r, g, b]) => [r, g, b, alpha]));
}

describe("light and color adjustments", () => {
  it.each(CASES)("makes the server's pixels when %s", (_name, sliders, expected) => {
    const adjusted = adjustPixels(rgba(PIXELS), PIXELS.length, { ...NEUTRAL_ADJUSTMENTS, ...sliders });

    expect(Array.from(adjusted)).toEqual(Array.from(rgba(expected)));
  });

  it("changes nothing with every slider at zero", () => {
    expect(isNeutral(NEUTRAL_ADJUSTMENTS)).toBe(true);
    for (const table of channelTables(NEUTRAL_ADJUSTMENTS)) {
      expect(Array.from(table)).toEqual(Array.from({ length: 256 }, (_, value) => value));
    }
  });

  it("leaves transparency as it was", () => {
    const pixels = new Uint8ClampedArray([200, 100, 50, 0, 200, 100, 50, 77]);

    const adjusted = adjustPixels(pixels, 2, { ...NEUTRAL_ADJUSTMENTS, brightness: 60, saturation: -50 });

    expect([adjusted[3], adjusted[7]]).toEqual([0, 77]);
  });
});

describe("shadows and highlights", () => {
  it("move the dark tones most for shadows and the light ones for highlights", () => {
    // 85 is a third of white and 170 two thirds. Shadows at 100 adds
    // 255 * 1/3 * (2/3)^2 = 37.8 to 85 and 255 * 2/3 * (1/3)^2 = 18.9 to 170;
    // highlights at -100 takes the same amounts the other way round.
    const lifted = channelTables({ ...NEUTRAL_ADJUSTMENTS, shadows: 100 })[1];
    const recovered = channelTables({ ...NEUTRAL_ADJUSTMENTS, highlights: -100 })[1];

    expect([0, 85, 170, 255].map((level) => lifted[level])).toEqual([0, 123, 189, 255]);
    expect([0, 85, 170, 255].map((level) => recovered[level])).toEqual([0, 66, 132, 255]);
  });

  it("never swap two levels, however the two are set", () => {
    for (const shadows of [-100, -37, 0, 64, 100]) {
      for (const highlights of [-100, -51, 0, 23, 100]) {
        for (const table of channelTables({ ...NEUTRAL_ADJUSTMENTS, shadows, highlights })) {
          expect([table[0], table[255]]).toEqual([0, 255]);
          expect(table.every((level, index) => index === 0 || table[index - 1] <= level)).toBe(true);
        }
      }
    }
  });

  it("each count as a change on their own", () => {
    expect(isNeutral({ ...NEUTRAL_ADJUSTMENTS, highlights: -1 })).toBe(false);
    expect(isNeutral({ ...NEUTRAL_ADJUSTMENTS, shadows: 1 })).toBe(false);
  });
});

describe("whites and blacks", () => {
  it("move white and black a quarter of the range at most, as the server's tables do", () => {
    // The same levels are pinned in test_studio_adjustments.py.
    const at = (sliders: Partial<StudioColorAdjustments>, levels: number[]) =>
      levels.map((level) => channelTables({ ...NEUTRAL_ADJUSTMENTS, ...sliders })[1][level]);
    expect(at({ blacks: 100 }, [0, 128, 255])).toEqual([64, 160, 255]);
    expect(at({ blacks: -100 }, [64, 65, 255])).toEqual([0, 2, 255]);
    expect(at({ whites: 100 }, [0, 190, 192])).toEqual([0, 253, 255]);
    expect(at({ whites: -100 }, [0, 128, 255])).toEqual([0, 96, 191]);
    expect(levelEnds(-100, 100)).toEqual([63.75, 0, 191.25, 255]);
  });

  it("hold white and black however far the other sliders take the picture", () => {
    const glaring = channelTables({ ...NEUTRAL_ADJUSTMENTS, brightness: 100, contrast: 100, whites: -100 });
    const murky = channelTables({ ...NEUTRAL_ADJUSTMENTS, brightness: -100, contrast: 100, blacks: 100 });

    expect(glaring.map((table) => Math.max(...table))).toEqual([191, 191, 191]);
    expect(murky.map((table) => Math.min(...table))).toEqual([64, 64, 64]);
  });

  it("each count as a change on their own", () => {
    expect(isNeutral({ ...NEUTRAL_ADJUSTMENTS, whites: -1 })).toBe(false);
    expect(isNeutral({ ...NEUTRAL_ADJUSTMENTS, blacks: 1 })).toBe(false);
  });
});

describe("sharpness", () => {
  it.each(SHARP_CASES)("makes the server's pixels for %s", (_name, sliders, cutout, expected) => {
    const pixels = new Uint8ClampedArray(
      SHARP_COLORS.flatMap(([r, g, b], index) => [r, g, b, cutout ? SHARP_ALPHAS[index] : 255]),
    );

    const adjusted = adjustPixels(pixels, SHARP_WIDTH, { ...NEUTRAL_ADJUSTMENTS, ...sliders });

    expect(Array.from(adjusted)).toEqual(expected.flat());
  });

  it("counts as a change on its own", () => {
    expect(isNeutral({ ...NEUTRAL_ADJUSTMENTS, sharpness: -1 })).toBe(false);
  });
});

describe("vibrance and the vignette", () => {
  it.each(VIGNETTE_CASES)("make the server's pixels for %s", (_name, sliders, cutout, expected) => {
    const pixels = new Uint8ClampedArray(
      SHARP_COLORS.flatMap(([r, g, b], index) => [r, g, b, cutout ? SHARP_ALPHAS[index] : 255]),
    );

    const adjusted = adjustPixels(pixels, SHARP_WIDTH, { ...NEUTRAL_ADJUSTMENTS, ...sliders });

    expect(Array.from(adjusted)).toEqual(expected.flat());
  });

  it("each count as a change on their own", () => {
    expect(isNeutral({ ...NEUTRAL_ADJUSTMENTS, vibrance: -1 })).toBe(false);
    expect(isNeutral({ ...NEUTRAL_ADJUSTMENTS, vignette: 1 })).toBe(false);
  });

  it("leave the middle of the picture and gather evenly toward the corners", () => {
    const width = 64;
    const height = 48;
    const grey = new Uint8ClampedArray(width * height * 4).fill(160);

    const adjusted = adjustPixels(grey, width, { ...NEUTRAL_ADJUSTMENTS, vignette: 100 });
    const red = (x: number, y: number) => adjusted[(y * width + x) * 4];

    expect([red(31, 23), red(32, 24)]).toEqual([160, 160]);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        // Mirrored left to right and top to bottom, as a centred vignette is.
        expect([red(width - 1 - x, y), red(x, height - 1 - y)]).toEqual([red(x, y), red(x, y)]);
      }
    }
    expect(red(0, 0)).toBeLessThan(red(0, 24));
    expect(red(0, 24)).toBeLessThan(160);
  });

  it("take the vignette on a picture one pixel wide or tall", () => {
    const strip = new Uint8ClampedArray(3 * 4).fill(160);

    const tall = adjustPixels(strip, 1, { ...NEUTRAL_ADJUSTMENTS, vignette: 100 });
    const wide = adjustPixels(strip, 3, { ...NEUTRAL_ADJUSTMENTS, vignette: 100 });

    expect([tall[0], tall[4], tall[8]]).toEqual([154, 160, 154]);
    expect(Array.from(wide)).toEqual(Array.from(tall));
  });
});

describe("grain", () => {
  it.each(GRAIN_CASES)("makes the server's pixels for %s", (_name, sliders, cutout, expected) => {
    const pixels = new Uint8ClampedArray(
      SHARP_COLORS.flatMap(([r, g, b], index) => [r, g, b, cutout ? SHARP_ALPHAS[index] : 255]),
    );

    const adjusted = adjustPixels(pixels, SHARP_WIDTH, { ...NEUTRAL_ADJUSTMENTS, ...sliders });

    expect(Array.from(adjusted)).toEqual(expected.flat());
  });

  it("takes the server's levels from the same whole-number hash", () => {
    // The same numbers are pinned in test_studio_adjustments.py.
    expect(Array.from({ length: 8 }, (_, x) => grainLevel(x, 0))).toEqual([59, 133, 151, 193, 232, 132, 185, 110]);
    expect([grainLevel(255, 255), grainLevel(3, 7)]).toEqual([126, 172]);
  });

  it("moves all three channels alike, repeats every tile, and keeps the picture's light", () => {
    const width = 512;
    const height = 8;
    const grey = new Uint8ClampedArray(width * height * 4).fill(128);

    const grained = adjustPixels(grey, width, { ...NEUTRAL_ADJUSTMENTS, grain: 100 });
    let sum = 0;
    for (let pixel = 0; pixel < width * height; pixel += 1) {
      const at = pixel * 4;
      expect([grained[at + 1], grained[at + 2]]).toEqual([grained[at], grained[at]]);
      expect(grained[at]).toBe(grained[(pixel % width < 256 ? pixel + 256 : pixel - 256) * 4]);
      sum += grained[at];
    }
    expect(Math.abs(sum / (width * height) - 128)).toBeLessThan(1);
    expect(grainOffsets(100)[255] - 128).toBeLessThanOrEqual(32);
  });

  it("counts as a change on its own", () => {
    expect(isNeutral({ ...NEUTRAL_ADJUSTMENTS, grain: 1 })).toBe(false);
  });
});
