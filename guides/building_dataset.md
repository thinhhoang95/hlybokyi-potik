# Important
> please ignore the pre_scripts folder. They are legacy code and already deprecated.
> Use pre_scripts2 code instead.

- Use `pre_scripts2/1_1download_ostrino.py` to download ADS-B data from OpenSky.
- Then, use `create_merge_instrs_server.ipynb` to merge the dangling flights (flights existing between two hourly CSV files).
- The script to download weather file is `download_grib.ipynb`.
- Use the `pre_scripts2/get_route.py` to extract the flown routes. However, if it's too slow on the local machine, hire a runpod instance with around 150GB of data storage (very important), transfer the cs.zip containing the cs directory remotely using:
```
rsync -av --progress --partial --inplace -e "ssh -p 19200" cs.zip root@154.54.102.18:/root/hlybokyi-potik/cs.zip
```
This is helpful for resuming if on flaky links.
To download, use `tar czvf` to compress the output files, and `rsync` again:
```
rsync -av --progress --partial --inplace -e "ssh -p 19200" root@154.54.102.18:/root/hlybokyi-potik/summer24/routes_fast/cs_2024.tar.gz ./cs_2024.tar.gz
```
- After building the segment dataset, verify a few IDs with the notebook `inspect_routes.ipynb` and compare the segments to the original ADS-B dataset.
- Use `count_blocks.ipynb` notebook to build a basic count output.
- Use `infer_route_auto_server.py` script from project-akrav (full path: `project-akrav/route_infer/infer_route_auto_server.py`) (route_infer module) to obtain the waypoint named routes. Use the `infer_route52` since it supports 4D trajectory output (instead of just 3D).