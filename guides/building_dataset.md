# Important
> please ignore the pre_scripts folder. They are legacy code and already deprecated.
> Use pre_scripts2 code instead.

- Use `pre_scripts2/1_1download_ostrino.py` to download ADS-B data from OpenSky.
- Then, use `create_merge_instrs_server.ipynb` to merge the dangling flights (flights existing between two hourly CSV files).
- The script to download weather file is `download_grib.ipynb`.
- Use the `pre_scripts2/get_route.py` to extract the flown routes.
- Use `count_blocks.ipynb` notebook to build a basic count output.
- Use `infer_route_auto_server.py` script from project-akrav (route_infer module) to obtain the waypoint named routes.