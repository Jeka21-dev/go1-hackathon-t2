# Образ строится поверх образа Simulator: в нём есть ROS 2 Humble и unitree_go.
ARG SIM_IMAGE=registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0
FROM ${SIM_IMAGE}
USER root
RUN python3 -c "import mujoco, numpy" 2>/dev/null || pip3 install --no-cache-dir mujoco numpy
COPY safety_block.py selftest.py collision_test.py /opt/t2/
COPY unitree_go1 /opt/t2/unitree_go1
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
USER ros
ENTRYPOINT ["/entrypoint.sh"]
CMD ["T2"]
