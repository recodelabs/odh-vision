#!/usr/bin/env python3
"""
2_enhance_regions.py  —  Crop and enhance scan regions for OCR / re-reading.

Usage:
    python 2_enhance_regions.py <image> [--box L,T,R,B] [--scale N] [--mode contrast|threshold]

Examples:
    # Enhance the name column of page 3
    python 2_enhance_regions.py _output/Apr_week_2_p3.jpg --box 770,150,1560,3300

    # Binary threshold the whole image
    python 2_enhance_regions.py _output/page.jpg --mode threshold --scale 3
"""

import sys, os, argparse
from extraction_helpers import enhance_region


def main():
    p = argparse.ArgumentParser(description="Enhance a scan region.")
    p.add_argument("image")
    p.add_argument("--box", type=str, default=None,
                   help="Crop box L,T,R,B pixels")
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--mode", choices=["contrast", "threshold"], default="contrast")
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args()

    if not os.path.isfile(args.image):
        print(f"ERROR: not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    box = None
    if args.box:
        parts = [int(x) for x in args.box.split(",")]
        if len(parts) != 4:
            print("ERROR: --box needs 4 values L,T,R,B", file=sys.stderr)
            sys.exit(1)
        box = tuple(parts)

    out = args.output or (
        os.path.splitext(args.image)[0] + "_enhanced"
        + os.path.splitext(args.image)[1])

    result = enhance_region(args.image, out, box=box,
                            scale=args.scale, mode=args.mode)
    print(f"Saved → {result}")


if __name__ == "__main__":
    main()
