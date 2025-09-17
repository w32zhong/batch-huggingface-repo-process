#!/usr/bin/env python3
"""Batch utility to modify Hugging Face repository ownership or visibility."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Callable, Dict, Iterable, Union

from huggingface_hub import HfApi
from huggingface_hub.hf_api import DatasetInfo, ModelInfo, SpaceInfo
from huggingface_hub.utils import HfHubHTTPError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch process Hugging Face repositories for a user: either set all public repos to private"
            " or transfer private repos to another owner."
        )
    )
    parser.add_argument(
        "--user",
        default="w32zhong",
        help="Hugging Face username or organization whose repositories will be updated.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Hugging Face access token with admin rights on the target repos.\n"
            "If omitted, the script falls back to the HF_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--repo-types",
        nargs="+",
        default=("model", "dataset", "space"),
        choices=("model", "dataset", "space"),
        help="Repository types to scan. Default checks models, datasets, and spaces.",
    )
    parser.add_argument(
        "--operation",
        choices=("set-private", "transfer"),
        default="transfer",
        help=(
            "Which batch operation to perform. `set-private` converts all public repos to private;"
            " `transfer` moves private repos to another owner. Default: transfer."
        ),
    )
    parser.add_argument(
        "--target",
        default="TAIL-LGE-HF",
        help=(
            "New owner for repositories when using the transfer operation. Ignored for set-private."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report repositories that would be changed without applying updates.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity for troubleshooting.",
    )
    return parser.parse_args()


def get_token(explicit_token: str | None) -> str:
    token = explicit_token or os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "A Hugging Face token is required. Provide it with --token or set HF_TOKEN."
        )
    return token


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


RepoInfoType = Union[ModelInfo, DatasetInfo, SpaceInfo]
RepoFetcher = Callable[[HfApi, str], Iterable[RepoInfoType]]


FETCHERS: Dict[str, RepoFetcher] = {
    "model": lambda api, user: api.list_models(author=user, full=True, limit=None),
    "dataset": lambda api, user: api.list_datasets(author=user, full=True, limit=None),
    "space": lambda api, user: api.list_spaces(author=user, full=True, limit=None),
}


def collect_public_repos(
    api: HfApi, user: str, repo_types: Iterable[str]
) -> tuple[list[tuple[str, RepoInfoType]], bool]:
    public_repos: list[tuple[str, RepoInfoType]] = []
    had_errors = False
    for repo_type in repo_types:
        fetcher = FETCHERS[repo_type]
        logging.debug("Fetching %s repositories for %s", repo_type, user)
        try:
            for repo in fetcher(api, user):
                if not getattr(repo, "private", False):
                    public_repos.append((repo_type, repo))
        except HfHubHTTPError as err:
            logging.error("Failed to list %s repositories for %s: %s", repo_type, user, err)
            had_errors = True
        except Exception as err:  # pragma: no cover - defensive fallback
            logging.error("Unexpected error while listing %s repositories: %s", repo_type, err)
            had_errors = True
    return public_repos, had_errors


def collect_private_repos(
    api: HfApi, user: str, repo_types: Iterable[str]
) -> tuple[list[tuple[str, RepoInfoType]], bool]:
    private_repos: list[tuple[str, RepoInfoType]] = []
    had_errors = False
    for repo_type in repo_types:
        fetcher = FETCHERS[repo_type]
        logging.debug("Fetching %s repositories for %s", repo_type, user)
        try:
            for repo in fetcher(api, user):
                if getattr(repo, "private", False):
                    private_repos.append((repo_type, repo))
        except HfHubHTTPError as err:
            logging.error("Failed to list %s repositories for %s: %s", repo_type, user, err)
            had_errors = True
        except Exception as err:  # pragma: no cover - defensive fallback
            logging.error("Unexpected error while listing %s repositories: %s", repo_type, err)
            had_errors = True
    return private_repos, had_errors


def set_private(
    api: HfApi,
    repo_type: str,
    repo: RepoInfoType,
    source_user: str,
    dry_run: bool,
) -> None:
    repo_id = (
        getattr(repo, "repo_id", None)
        or getattr(repo, "id", None)
        or getattr(repo, "modelId", None)
        or getattr(repo, "name", None)
    )
    if not repo_id:
        logging.error(
            "Skipping %s repository with unknown identifier: %s", repo_type, repr(repo)
        )
        return
    if "/" not in repo_id:
        repo_id = f"{source_user}/{repo_id}"
    if getattr(repo, "private", False):
        logging.debug("%s: %s already private; skipping", repo_type, repo_id)
        return

    logging.info("%s: %s -> set private", repo_type, repo_id)
    if dry_run:
        return

    try:
        api.update_repo_visibility(repo_id=repo_id, private=True, repo_type=repo_type)
    except HfHubHTTPError as err:
        logging.error("Failed to set private for %s (%s): %s", repo_id, repo_type, err)
    else:
        logging.debug("Successfully set private %s (%s)", repo_id, repo_type)


def transfer_repo(
    api: HfApi,
    repo_type: str,
    repo: RepoInfoType,
    target: str,
    source_user: str,
    dry_run: bool,
) -> None:
    repo_id = (
        getattr(repo, "repo_id", None)
        or getattr(repo, "id", None)
        or getattr(repo, "modelId", None)
        or getattr(repo, "name", None)
    )
    if not repo_id:
        logging.error(
            "Skipping %s repository with unknown identifier: %s", repo_type, repr(repo)
        )
        return
    if "/" not in repo_id:
        repo_id = f"{source_user}/{repo_id}"
    owner = repo_id.split("/", 1)[0]
    if owner and owner == target:
        logging.info("%s: %s already owned by %s; skipping", repo_type, repo_id, target)
        return

    _, repo_name = repo_id.split("/", 1)
    destination = f"{target}/{repo_name}"

    logging.info("%s: %s -> transfer to %s", repo_type, repo_id, destination)
    if dry_run:
        return
    try:
        api.move_repo(from_id=repo_id, to_id=destination, repo_type=repo_type)
    except HfHubHTTPError as err:
        logging.error("Failed to transfer %s (%s): %s", repo_id, repo_type, err)
    else:
        logging.debug("Successfully transferred %s (%s)", repo_id, repo_type)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    try:
        token = get_token(args.token)
    except SystemExit as exc:
        logging.error(exc)
        return 1

    api = HfApi(token=token)

    if args.operation == "set-private":
        repos, had_errors = collect_public_repos(api, args.user, args.repo_types)
        action = "set private"
        for repo_type, repo in repos:
            set_private(api, repo_type, repo, args.user, args.dry_run)
    else:
        repos, had_errors = collect_private_repos(api, args.user, args.repo_types)
        action = "transferred"
        if not args.target:
            logging.error("--target is required when using the transfer operation.")
            return 1
        for repo_type, repo in repos:
            transfer_repo(api, repo_type, repo, args.target, args.user, args.dry_run)

    changes = len(repos)

    if changes == 0:
        if had_errors:
            logging.warning(
                "Encountered errors while listing repositories; no updates were performed."
            )
            return 2
        logging.info(
            "No matching repositories found for %s during %s operation.",
            args.user,
            args.operation,
        )
        return 0

    if args.dry_run:
        logging.info("Dry run complete. %d repositories would be %s.", changes, action)
    else:
        logging.info("Successfully %s %d repositories.", action, changes)

    if had_errors:
        logging.warning("Completed with errors; review the log output above.")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
