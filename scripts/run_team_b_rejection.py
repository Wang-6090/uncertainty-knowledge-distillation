"""Executor for the team-B research line (閺夊孩鐓版０? 閺堫亞鐓℃搴ㄦ珦娴兼媽顓告稉搴♀偓娆撯偓澶岀摣闁?.

Implements section 6 of ``docs/parallel_work_division.md``:

* 6.2 fixed risk score (``normalized_entropy_mahalanobis``) + known-validation
  threshold, per seed;
* 6.3 paired comparison of the two rejector configurations
  (``scratch_cnn`` vs ``shared_backbone``) with everything else fixed;
* writes the artifacts of section 9 (``config.json`` / ``command.txt`` /
  ``metrics.json`` / ``summary.md``) and the per-seed + multi-seed analysis of
  sections 6.2, 6.4 and 6.5.

Nothing here uses unknown labels for training, threshold calibration or sample
screening: unknown labels are read only by the retrospective analysis scripts.

Example (pilot, CPU friendly):

    python scripts/run_team_b_rejection.py --mode pilot --tag pilot_20261123

Example (formal protocol, three seeds, both rejector configurations):

    python scripts/run_team_b_rejection.py --mode formal --tag formal_20261124
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    REPO_ROOT.parent / "uncertainty-knowledge-distillation1.0" / "CIFAR-100-dataset-main"
)
DEFAULT_SPLIT = REPO_ROOT / "splits" / "cifar100_60_40_imagefolder.json"
FINE_NAMES_SOURCE = REPO_ROOT / "splits" / "protocols" / "cifar100_60_40_random.json"

MODES = ("scratch_cnn", "shared_backbone")


def log(message: str) -> None:
    stamp = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def build_common(args) -> list[str]:
    command = [
        "--dataset", args.dataset,
        "--data-root", str(args.data_root),
        "--split-path", str(args.split_path),
        "--num-known", str(args.num_known),
        "--image-size", str(args.image_size),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
    ]
    if args.limit_train:
        command += ["--limit-train", str(args.limit_train)]
    return command


def build_teacher_command(args, seed: int, work_dir: Path) -> list[str]:
    command = [
        sys.executable, "train.py", "train_teacher",
        *build_common(args),
        "--seed", str(seed),
        "--backbone", args.teacher_backbone,
        "--epochs", str(args.teacher_epochs),
        "--lr", str(args.lr),
        "--work-dir", str(work_dir),
    ]
    if args.pretrained:
        command.append("--pretrained")
    return command


def build_student_command(args, seed: int, mode: str, teacher_ckpt: Path,
                          work_dir: Path) -> list[str]:
    command = [
        sys.executable, "train.py", "train_student",
        *build_common(args),
        "--seed", str(seed),
        "--backbone", args.student_backbone,
        "--teacher-backbone", args.teacher_backbone,
        "--student-backbone", args.student_backbone,
        "--epochs", str(args.student_epochs),
        "--lr", str(args.lr),
        "--alpha-unc", str(args.alpha_unc),
        "--teacher-ckpt", str(teacher_ckpt),
        "--independent-rejection-encoder",
        "--independent-rejection-encoder-init", mode,
        "--rejection-feature-dim", str(args.rejection_feature_dim),
        "--work-dir", str(work_dir),
    ]
    if args.pretrained:
        command.append("--pretrained")
    if args.save_epoch_checkpoints:
        command.append("--save-epoch-checkpoints")
    return command


def build_discover_command(args, seed: int, mode: str, student_ckpt: Path,
                           work_dir: Path) -> list[str]:
    return [
        sys.executable, "train.py", "discover",
        *build_common(args),
        "--seed", str(seed),
        "--backbone", args.student_backbone,
        "--teacher-backbone", args.teacher_backbone,
        "--student-backbone", args.student_backbone,
        "--student-ckpt", str(student_ckpt),
        "--independent-rejection-encoder",
        "--independent-rejection-encoder-init", mode,
        "--rejection-feature-dim", str(args.rejection_feature_dim),
        "--score-mode", args.score_mode,
        "--threshold-policy", args.threshold_policy,
        "--threshold-percentile", str(args.threshold_percentile),
        "--mc-samples", str(args.mc_samples),
        "--cluster-k", args.cluster_k,
        "--cluster-feature", args.cluster_feature,
        "--cluster-method", args.cluster_method,
        "--save-calibration-scores",
        "--work-dir", str(work_dir),
    ]


def run_command(command: list[str], run_dir: Path, dry_run: bool,
                env: dict | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    command_path = run_dir / "command.txt"
    command_path.write_text(" ".join(command) + "\n", encoding="utf-8")
    log_path = run_dir / "train.log"
    if dry_run:
        log("DRY-RUN: " + " ".join(command))
        return log_path
    log(f"RUN ({run_dir.name}): " + " ".join(command[2:5]) + " ...")
    import os

    merged_env = os.environ.copy()
    if env:
        merged_env.update({key: str(value) for key, value in env.items()})
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=merged_env,
        )
    if completed.returncode != 0:
        tail = ""
        if log_path.is_file():
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        raise SystemExit(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"--- tail of {log_path} ---\n{tail}"
        )
    return log_path


def run_threshold_stability(discover_dir: Path, dry_run: bool) -> None:
    scores = discover_dir / "calibration_scores.npz"
    if not scores.is_file():
        log(f"skip threshold stability: {scores} missing")
        return
    command = [
        sys.executable, "scripts/audit_validation_threshold_stability.py", str(scores),
        "--target", "0.95",
        "--repeats", "2000",
        "--output", str(discover_dir / "threshold_stability.json"),
    ]
    if dry_run:
        log("DRY-RUN: " + " ".join(command))
        return
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def run_pool_analysis(args, run_root: Path, analysis_dir: Path, seeds: list[int],
                      dry_run: bool) -> None:
    for mode in args.init_modes:
        run_dirs = [
            run_root / f"s{seed}_{mode}_discover"
            for seed in seeds
            if (run_root / f"s{seed}_{mode}_discover").is_dir()
        ]
        if not run_dirs:
            continue
        labels = [f"s{seed}_{mode}" for seed in seeds if (run_root / f"s{seed}_{mode}_discover").is_dir()]
        command = [
            sys.executable, "scripts/analyze_team_b_candidate_pool.py",
            "--runs", *[str(path) for path in run_dirs],
            "--labels", *labels,
            "--split", str(args.split_path),
            "--fine-names", str(FINE_NAMES_SOURCE),
            "--output", str(analysis_dir / f"candidate_pool_{mode}.json"),
            "--markdown", str(analysis_dir / f"candidate_pool_{mode}.md"),
        ]
        if dry_run:
            log("DRY-RUN: " + " ".join(command))
            continue
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def run_summary(args, run_root: Path, analysis_dir: Path, dry_run: bool) -> None:
    command = [
        sys.executable, "scripts/summarize_team_b_rejection.py",
        "--work-root", str(run_root),
        "--output-dir", str(analysis_dir),
        "--split", str(args.split_path),
    ]
    if dry_run:
        log("DRY-RUN: " + " ".join(command))
        return
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--tag", default=None,
                        help="output sub-directory name; defaults to <mode>_<YYYYMMDD>")
    parser.add_argument("--seeds", default=None,
                        help="comma separated seeds; pilot defaults to 42,123, "
                             "formal defaults to 42,123,3407")
    parser.add_argument("--init-modes", default=",".join(MODES))
    parser.add_argument("--dataset", default="imagefolder")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--num-known", type=int, default=60)
    parser.add_argument("--teacher-backbone", default="resnet34")
    parser.add_argument("--student-backbone", default="resnet18")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-unc", type=float, default=0.1)
    parser.add_argument("--teacher-epochs", type=int, default=None)
    parser.add_argument("--student-epochs", type=int, default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rejection-feature-dim", type=int, default=64,
                        help="rejection embedding width; must be > 0 when the "
                             "independent rejection encoder is enabled")
    parser.add_argument("--score-mode", default="normalized_entropy_mahalanobis")
    parser.add_argument("--threshold-policy", default="global")
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--cluster-k", default="oracle")
    parser.add_argument("--cluster-feature", default="feature_pca")
    parser.add_argument("--cluster-method", default="kmeans")
    parser.add_argument("--work-root", default="runs/team_b_rejection")
    parser.add_argument("--analysis-root", default="analysis/team_b_rejection")
    parser.add_argument("--shared-teacher", action="store_true",
                        help="train the teacher once (first seed) and reuse it for the "
                             "remaining student seeds; pilot only")
    parser.add_argument("--skip-existing", action="store_true",
                        help="reuse an existing checkpoint instead of retraining it")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="cap the known training subset; 0 uses everything. "
                             "Pilot default 8000, formal default 0")
    parser.add_argument("--threads", type=int, default=8,
                        help="OMP/MKL thread count handed to the training subprocesses")
    parser.add_argument("--save-epoch-checkpoints", action="store_true",
                        help="ask train_student to dump student_epoch_<n>.pt every epoch "
                             "(crash resilience)")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="do not run the candidate-pool / summary pass at the end; "
                             "use when several processes share one output directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.image_size is None:
        args.image_size = 32 if args.mode == "pilot" else 64
    if args.teacher_epochs is None:
        args.teacher_epochs = 2 if args.mode == "pilot" else 60
    if args.student_epochs is None:
        args.student_epochs = 2 if args.mode == "pilot" else 60
    if args.seeds is None:
        args.seeds = "42,123" if args.mode == "pilot" else "42,123,3407"
    if args.limit_train is None:
        args.limit_train = 8000 if args.mode == "pilot" else 0
    if args.mc_samples <= 0:
        raise SystemExit("--mc-samples must be positive")
    thread_env = {}
    if args.threads > 0:
        thread_env = {
            "OMP_NUM_THREADS": args.threads,
            "MKL_NUM_THREADS": args.threads,
        }

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    args.init_modes = [value for value in args.init_modes.split(",") if value.strip()]
    unknown_modes = [mode for mode in args.init_modes if mode not in MODES]
    if unknown_modes:
        raise SystemExit(f"unsupported --init-modes: {unknown_modes}")

    args.data_root = Path(args.data_root)
    args.split_path = Path(args.split_path)
    if not args.data_root.is_dir():
        raise SystemExit(f"data root not found: {args.data_root}")
    if not args.split_path.is_file():
        raise SystemExit(f"split file not found: {args.split_path}")
    if args.split_path.resolve() != DEFAULT_SPLIT.resolve():
        log("WARNING: split file differs from the default imagefolder 60/40 protocol")

    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    log(f"split: known={len(split['known_classes'])}, novel={len(split['novel_classes'])}")

    tag = args.tag or f"{args.mode}_{_dt.date.today():%Y%m%d}"
    run_root = (REPO_ROOT / args.work_root / tag)
    analysis_dir = (REPO_ROOT / args.analysis_root / tag)
    run_root.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tag": tag,
        "mode": args.mode,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds,
        "init_modes": args.init_modes,
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "split_path": str(args.split_path),
        "num_known": args.num_known,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "teacher_backbone": args.teacher_backbone,
        "student_backbone": args.student_backbone,
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "score_mode": args.score_mode,
        "threshold_policy": args.threshold_policy,
        "threshold_percentile": args.threshold_percentile,
        "mc_samples": args.mc_samples,
        "rejection_feature_dim": args.rejection_feature_dim,
        "limit_train": args.limit_train,
        "threads": args.threads,
        "cluster": {
            "k": args.cluster_k,
            "feature": args.cluster_feature,
            "method": args.cluster_method,
        },
        "shared_teacher": args.shared_teacher,
        "note": (
            "Unknown labels are used only for retrospective evaluation; they never "
            "enter training, threshold calibration or sample screening."
        ),
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(f"run root: {run_root}")
    log(f"analysis: {analysis_dir}")

    teacher_checkpoints: dict[int, Path] = {}
    for index, seed in enumerate(seeds):
        teacher_dir = run_root / f"teacher_s{seed}"
        teacher_ckpt = teacher_dir / "teacher.pt"
        if args.shared_teacher and index > 0:
            source = teacher_checkpoints[seeds[0]]
            teacher_checkpoints[seed] = source
            log(f"seed {seed}: reuse shared teacher {source}")
            continue
        if teacher_ckpt.is_file() and args.skip_existing:
            log(f"seed {seed}: teacher exists, skip")
            teacher_checkpoints[seed] = teacher_ckpt
            continue
        run_command(build_teacher_command(args, seed, teacher_dir), teacher_dir, args.dry_run, thread_env)
        teacher_checkpoints[seed] = teacher_ckpt

    for seed in seeds:
        teacher_dir = run_root / f"teacher_s{seed}"
        teacher_ckpt = teacher_checkpoints.get(seed)
        if teacher_ckpt is None or (not args.dry_run and not teacher_ckpt.is_file()):
            teacher_ckpt = run_root / f"teacher_s{seeds[0]}" / "teacher.pt"
        log(f"seed {seed}: teacher = {teacher_ckpt}")
        for mode in args.init_modes:
            student_dir = run_root / f"s{seed}_{mode}_student"
            student_ckpt = student_dir / "student.pt"
            if not (student_ckpt.is_file() and args.skip_existing):
                run_command(
                    build_student_command(args, seed, mode, teacher_ckpt, student_dir),
                    student_dir,
                    args.dry_run,
                    thread_env,
                )
            else:
                log(f"seed {seed}/{mode}: student exists, skip")

            discover_dir = run_root / f"s{seed}_{mode}_discover"
            report = discover_dir / "discovery_report.json"
            if not (report.is_file() and args.skip_existing):
                run_command(
                    build_discover_command(args, seed, mode, student_ckpt, discover_dir),
                    discover_dir,
                    args.dry_run,
                    thread_env,
                )
            else:
                log(f"seed {seed}/{mode}: discovery exists, skip")
            run_threshold_stability(discover_dir, args.dry_run)

    if args.skip_analysis:
        log("skip analysis (--skip-analysis)")
        return 0
    run_pool_analysis(args, run_root, analysis_dir, seeds, args.dry_run)
    run_summary(args, run_root, analysis_dir, args.dry_run)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

