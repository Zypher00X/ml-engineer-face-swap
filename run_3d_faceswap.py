"""CLI for portrait transfer. Legacy implementation is archived in experiments/faceswap."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Transfer source identity to target portrait pose and expression.')
    parser.add_argument('-s', '--source', default='img35.jpg')
    parser.add_argument('-t', '--target', default='img37.jpg')
    parser.add_argument('-o', '--output', default='outputs/portrait.png')
    parser.add_argument('--source-index', type=int, default=0, help='Index sorted by descending face area')
    parser.add_argument('--target-index', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--color-strength', type=float, default=0.5)
    parser.add_argument('--feather-ratio', type=float, default=0.035)
    parser.add_argument('--stitching', action='store_true')
    parser.add_argument('--no-retarget-lips', action='store_true')
    parser.add_argument('--backend',choices=['liveportrait','inswapper','hyperswap'],default='inswapper',help='hyperswap is an experimental native-256 backend')
    parser.add_argument('--input-sampling',choices=['native','light','balanced','area'],default='native',help='area reduces downsampling aliasing; changes model predictions and needs review')
    parser.add_argument('--protect-bridge',action='store_true',help='Experimental edge-based glasses bridge protection')
    parser.add_argument('--protection-mask',help='Optional target-sized grayscale mask: white preserves target pixels exactly')
    parser.add_argument('--scale-policy', choices=['source', 'target'], default='source')
    parser.add_argument('--glasses-policy',choices=['frames','full'],default='frames',
                        help='frames replaces transparent lens interiors; full restores legacy behavior')
    parser.add_argument('--legacy-color-transfer',action='store_true',help='Apply color transfer to eyes/lips too')
    parser.add_argument('--two-step-sampling',action='store_true',help='Use the legacy 128-to-512-to-output sampling')
    parser.add_argument('--debug',action='store_true',help='Save native model output, masks and intermediate crops')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    resolve = lambda p: Path(p) if Path(p).is_absolute() else root / p
    from faceswap.pipeline import Config, create_pipeline, read_image
    source, target, output = map(resolve, [args.source, args.target, args.output])
    read_image(source)
    read_image(target)
    config = Config(color_strength=args.color_strength, feather_ratio=args.feather_ratio,
                    stitching=args.stitching, scale_policy=args.scale_policy, device=args.device,
                    retarget_lips=not args.no_retarget_lips,backend=args.backend,
                    glasses_policy=args.glasses_policy,skin_color_only=not args.legacy_color_transfer,
                    direct_sampling=not args.two_step_sampling,save_debug=args.debug,
                    input_sampling=args.input_sampling,protect_bridge=args.protect_bridge,
                    protection_mask=str(resolve(args.protection_mask)) if args.protection_mask else None)
    _, _, meta = create_pipeline(config).swap(source, target, output, args.source_index, args.target_index)
    print(f"Saved {output} ({meta['seconds']:.2f} seconds, {meta['device']})")
    if not meta['quality']['project_gate_pass']:
        print(f"Review required: one or more project quality gates failed. Details: {output.with_suffix('.json')}")
    else:
        print(f"Automated geometry/identity gates passed; visual quality remains unverified. Details: {output.with_suffix('.json')}")
    if args.debug:
        print(f"Intermediate images: {meta['debug_directory']}")


if __name__ == '__main__':
    main()
