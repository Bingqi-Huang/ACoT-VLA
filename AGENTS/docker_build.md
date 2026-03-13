# Steps for building and testing submit-ready image

## Checkpoint
1. Copy checkpoint from trianing machine to local machine: Run following command on the training machine:
    ```bash
    rsync -avP -e "ssh -p 2222" /data/admins/bingqi/Projects/ACoT-VLA/checkpoints/acot_challenge_generalist_lora_generalist/generalist_v1_bs96/20000/params bingqi@101.6.33.98:/mnt/SharedData/Research/submit-ACoT-VLA-generalists-v1-20000/checkpoint/generalist-v1-20000/params
    ```
    ```bash
    rsync -avP -e "ssh -p 2222" /data/admins/bingqi/Projects/ACoT-VLA/checkpoints/acot_challenge_generalist_lora_generalist/generalist_v1_bs96/20000/assets bingqi@101.6.33.98:/mnt/SharedData/Research/submit-ACoT-VLA-generalists-v1-20000/checkpoint/generalist-v1-20000/assets
    ```
    ```bash
    rsync -avP -e "ssh -p 2222" /data/admins/bingqi/Projects/ACoT-VLA/checkpoints/acot_challenge_generalist_lora_generalist/generalist_v1_bs96/_CHECKPOINT_METADATA bingqi@101.6.33.98:/mnt/SharedData/Research/submit-ACoT-VLA-generalists-v1-20000/checkpoint/generalist-v1-20000/
    ```

2. Modify check point dir locally:
   - Change the docker file to copy correct checkpoint dir: **Change line 8** of `./scrips/docker/serve_policy_generalists_v1.Dockerfile`
   - Change the serve policy script to match correct checkpoint dir: **Change line 17** of `./scrips/docker/serve_submit_generalist.sh`

3. Build the image:
    ```bash
    docker build --no-cache   -f scripts/docker/serve_policy.generalists_v1.Dockerfile   -t generalists-v1-20000:latest   .
    ```

4. Run the image to test: 
    ```bash
    docker run -it --rm --network=host --gpus all   -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.3   generalists-v1-20000:latest
    ```
    **IMPORTANT: This should show no uv install or checkpoing download at all!**

5. Run the official ICRA test **offline**: 
   ```bash
   cd /mnt/SharedData/Research/genie_sim
   ./scripts/start_gui.sh
   ./scripts/into.sh
   ./scripts/run_icra_tasks.sh
   ```

6. **WAIT UNTIL THE RESULTS ARE SAVED**:
    ```bash
    # YOU SHOULD IN genie-sim folder or in the genie-sim container to run following cmds
    mv ./output/benchmark/ ./output/20000-benchmark/
    ```

7. **TAG THE IMAGE W.R.T official docs**:
    ```bash
    docker tag generalists-v1-20000:latest \
    sim-icra-registry.cn-beijing.cr.aliyuncs.com/onecable/generalists-v1-20000:latest
    ```

8. **PUSH THE IMAGE**:
   ```bash
   # Login with credentials
   # Check the website
   # Push
   docker push sim-icra-registry.cn-beijing.cr.aliyuncs.com/onecable/generalists-v1-20000:latest
   ```
