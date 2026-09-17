#!/bin/bash
# ------------------------------------------------------------------
# @File    :   download_checkpoint.sh
# @Time    :   2024/03/16 17:30:00
# @Author  :   Weihao Xia (xiawh3@outlook.com)
# @Version :   1.0
# @Desc    :   download Checkpoints from Hugging Face
# ------------------------------------------------------------------

set -euo pipefail

HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
REPO_ID="weihaox/umbrae"
TARGET_DIR="./"
USE_HF_HUB="${USE_HF_HUB:-0}"

echo "Using HF endpoint: ${HF_ENDPOINT}"

download_with_hf_hub() {
	HF_ENDPOINT="${HF_ENDPOINT}" python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
		repo_id="weihaox/umbrae",
		repo_type="dataset",
		local_dir="./",
		ignore_patterns=["all_images.pt", ".gitattributes"],
		max_workers=1,
)
PY
}

download_with_git() {
	local tmp_dir
	tmp_dir="$(mktemp -d .hf_dataset_tmp.XXXXXX)"
	local git_url
	local attempt
	local max_attempts
	max_attempts=3
	if [[ "${HF_ENDPOINT}" == *"hf-mirror.com"* ]]; then
		# hf-mirror currently resolves LFS files via cas-bridge.xethub.hf-mirror.org,
		# which is not reachable in some environments.
		git_url="https://huggingface.co/datasets/${REPO_ID}"
		echo "Detected hf-mirror endpoint; switching git+lfs source to: ${git_url}"
	else
		git_url="${HF_ENDPOINT%/}/datasets/${REPO_ID}"
	fi

	echo "Falling back to git clone: ${git_url}"
	echo "Using temporary directory: ${tmp_dir}"

	GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${git_url}" "${tmp_dir}"

	(
		cd "${tmp_dir}"
		attempt=1
		until git lfs pull --exclude "all_images.pt"; do
			if [ ${attempt} -ge ${max_attempts} ]; then
				echo "git lfs pull failed after ${max_attempts} attempts."
				return 1
			fi
			attempt=$((attempt + 1))
			echo "git lfs pull failed, retry ${attempt}/${max_attempts}..."
		done
		rm -f .gitattributes
	)

	# Copy downloaded files and avoid importing source .git metadata.
	mkdir -p "${TARGET_DIR}"
	tar --exclude='.git' -cf - -C "${tmp_dir}" . | tar -xf - -C "${TARGET_DIR}"

	# On NFS, temporary .nfs files can be busy; cleanup failure should not abort.
	rm -rf "${tmp_dir}" >/dev/null 2>&1 || true
}

if [ "${USE_HF_HUB}" = "1" ]; then
	set +e
	download_with_hf_hub
	status=$?
	set -e

	if [ ${status} -eq 0 ]; then
		echo "Checkpoint download completed via huggingface_hub."
		exit 0
	fi

	echo "huggingface_hub download failed (exit code: ${status})."
	echo "Trying git+lfs fallback..."
fi

download_with_git
echo "Checkpoint download completed via git+lfs."