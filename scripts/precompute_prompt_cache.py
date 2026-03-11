import pathlib

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import openpi.training.config as _config
import openpi.training.data_loader_fast as _data_loader_fast
import openpi.training.episode_split as _episode_split
import openpi.transforms as _transforms
import tyro


_SORT_PACKAGE_COLORS = ("white", "red", "black", "yellow")


def main(
    config_name: str,
    *,
    split: str = "train",
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    tokenize_transform = None
    for transform in data_config.model_transforms.inputs:
        if isinstance(transform, _transforms.TokenizePrompt):
            tokenize_transform = transform
            break
    if tokenize_transform is None:
        raise ValueError(f"Config `{config_name}` does not use TokenizePrompt.")

    prompt_strings = sorted(_collect_prompt_strings(config, data_config, split=split))
    tokens = []
    masks = []
    for prompt in prompt_strings:
        prompt_tokens, prompt_mask = tokenize_transform.tokenizer.tokenize(prompt)
        tokens.append(prompt_tokens)
        masks.append(prompt_mask)

    cache_path = _data_loader_fast.prompt_token_cache_path(config, split)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        prompts=np.asarray(prompt_strings, dtype=object),
        tokens=np.stack(tokens, axis=0),
        masks=np.stack(masks, axis=0),
    )
    print(f"Saved {len(prompt_strings)} prompt token entries to {cache_path}")


def _collect_prompt_strings(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    *,
    split: str,
) -> set[str]:
    selected_episodes = None
    if split != "all" and _episode_split.split_enabled(data_config):
        manifest, _ = _episode_split.get_or_create_manifest(
            data_config,
            base_output_dir=config.assets_dirs / "episode_splits",
        )
        selected_episodes = _episode_split.episodes_for_split(manifest, split)

    repo_ids = _episode_split.resolve_repo_ids(data_config.repo_id)
    prompts: set[str] = set()
    for repo_id in repo_ids:
        repo_path = pathlib.Path(repo_id).expanduser()
        if repo_path.is_absolute():
            metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_path.name, root=repo_path)
        else:
            metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

        allowed = None
        if isinstance(selected_episodes, dict):
            allowed = set(selected_episodes.get(repo_id, []))
        elif isinstance(selected_episodes, list):
            allowed = set(selected_episodes)

        segments = metadata.info.get("instruction_segments", {})
        for episode_key, episode_segments in segments.items():
            episode_index = int(episode_key)
            if allowed is not None and episode_index not in allowed:
                continue
            for segment in episode_segments:
                prompts.add(str(segment["instruction"]))

    prompt_map = {}
    for transform in data_config.data_transforms.inputs:
        prompt_map = getattr(transform, "prompt_map_inject_to_training", None) or {}
        if prompt_map:
            break
    for task_name, (prompt_template, _) in prompt_map.items():
        if task_name == "Sort packages":
            for color in _SORT_PACKAGE_COLORS:
                prompts.add(prompt_template.replace("<color>", color))
        else:
            prompts.add(prompt_template)
    return prompts


if __name__ == "__main__":
    tyro.cli(main)
