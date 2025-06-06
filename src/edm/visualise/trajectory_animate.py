# src/edm/visualise/trajectory_animate.py

import argparse
import re
from pathlib import Path
from typing import List

import imageio.v2 as imageio  # Ensure v2 for compatibility


def _int_from_name(path: Path) -> int:
    m = re.search(r"\d+", path.stem)
    return int(m.group()) if m else -1


def _collect_frames(folder: Path, pattern: str) -> List[Path]:
    files = sorted(folder.glob(pattern), key=_int_from_name)
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {folder}")
    return files


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dir", help="Directory with PNG frames")
    p.add_argument("--pattern", default="*.png", help="Glob pattern")
    p.add_argument("--fps", type=float, default=5, help="Frames per second")
    p.add_argument("--reverse", action="store_true",
                   help="Append reverse to the end (ping-pong effect)")
    p.add_argument("--backwards", action="store_true",
                   help="Play the animation backwards")
    p.add_argument("--mp4", action="store_true", help="Save as MP4")
    p.add_argument("--loop", type=int, default=0, help="GIF loop count")
    p.add_argument("--out", type=str, help="Output file name")
    return p.parse_args()


def save_gif(frames: List[Path], outfile: Path, fps: float, loop: int):
    imgs = [imageio.imread(f) for f in frames]
    imageio.mimsave(outfile, imgs, duration=1/fps, loop=loop)
    print(f"[saved] {outfile}  ({len(frames)} frames @ {fps} fps)")


def save_mp4(frames: List[Path], outfile: Path, fps: float):
    writer = imageio.get_writer(outfile, fps=fps, codec='libx264', quality=8)
    for f in frames:
        writer.append_data(imageio.imread(f))
    writer.close()
    print(f"[saved] {outfile}  ({len(frames)} frames @ {fps} fps)")


def main():
    args = parse_args()
    folder = Path(args.dir).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    frames = _collect_frames(folder, args.pattern)

    if args.backwards:
        frames = list(reversed(frames))

    if args.reverse:
        frames = frames + frames[-2:0:-1]  # forward + reverse (ping-pong)

    ext = ".mp4" if args.mp4 else ".gif"
    outfile = Path(args.out) if args.out else folder / f"trajectory{ext}"

    if args.mp4:
        save_mp4(frames, outfile, args.fps)
    else:
        save_gif(frames, outfile, args.fps, args.loop)


if __name__ == "__main__":
    main()




# # src/edm/visualise/trajectory_animate.py
# """
# Turn one time-series of molecule PNGs into an animation.

# Example
# -------
# python -m edm.visualise.trajectory_animate \
#        plots/trajectory/sample_00 --fps 6 --reverse
# """

# import argparse
# import re
# import subprocess
# from pathlib import Path
# from typing import List

# import imageio                      # <- v2 API is always present


# # --------------------------------------------------------------------------- #
# def _int_from_name(path: Path) -> int:
#     """Extract the first integer found in *path.name* (defaults to -1)."""
#     m = re.search(r"\d+", path.stem)
#     return int(m.group()) if m else -1


# def _collect_frames(folder: Path, pattern: str) -> List[Path]:
#     files = sorted(folder.glob(pattern), key=_int_from_name)
#     if not files:
#         raise FileNotFoundError(f"No files matching '{pattern}' in {folder}")
#     return files


# # --------------------------------------------------------------------------- #
# def parse_args():
#     p = argparse.ArgumentParser(
#         description="Concatenate trajectory PNGs into an animated GIF "
#                     "or optionally an MP4",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     p.add_argument("dir", metavar="DIR",
#                    help="Directory that contains 00.png, 0250.png, …")
#     p.add_argument("--pattern", default="*.png",
#                    help="Glob pattern to pick frames inside DIR")
#     p.add_argument("--fps", type=float, default=5,
#                    help="Frames per second")
#     p.add_argument("--reverse", action="store_true",
#                    help="Append the frames in reverse order for a ping-pong")
#     p.add_argument("--mp4", action="store_true",
#                    help="Write MP4 instead of GIF (needs ffmpeg on PATH)")
#     p.add_argument("--loop", type=int, default=0,
#                    help="Number of loops for GIF. 0 = infinite")
#     p.add_argument("--out", type=str, default=None,
#                    help="Output file name. Default: DIR/trajectory.(gif|mp4)")
#     return p.parse_args()


# # --------------------------------------------------------------------------- #
# def save_gif(frames: List[Path], outfile: Path, fps: float, loop: int):
#     duration = 1 / fps                       # seconds per frame
#     imgs = [imageio.imread(f) for f in frames]
#     imageio.mimsave(outfile, imgs, duration=duration, loop=loop)
#     print(f"[saved] {outfile}  ({len(frames)} frames @ {fps} fps)")


# def save_mp4(frames: List[Path], outfile: Path, fps: float):
#     ffmpeg_cmd = [
#         "ffmpeg", "-loglevel", "error", "-y",
#         "-f", "image2pipe", "-vcodec", "png", "-r", str(fps), "-i", "-",
#         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(outfile)
#     ]
#     proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
#     try:
#         for f in frames:
#             proc.stdin.write(imageio.imread(f).tobytes())
#     finally:
#         proc.stdin.close()
#         proc.wait()
#     print(f"[saved] {outfile}  ({len(frames)} frames @ {fps} fps)")


# # --------------------------------------------------------------------------- #
# def main():
#     args = parse_args()
#     folder = Path(args.dir).expanduser().resolve()
#     if not folder.is_dir():
#         raise NotADirectoryError(folder)

#     frames = _collect_frames(folder, args.pattern)

#     if args.reverse:
#         frames = frames + frames[-2:0:-1]            # forward + reverse loop

#     ext = ".mp4" if args.mp4 else ".gif"
#     outfile = Path(args.out) if args.out else folder / f"trajectory{ext}"

#     if args.mp4:
#         save_mp4(frames, outfile, args.fps)
#     else:
#         save_gif(frames, outfile, args.fps, args.loop)


# if __name__ == "__main__":
#     main()
