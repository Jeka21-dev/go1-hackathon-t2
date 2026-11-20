#!/usr/bin/env bash
# T1: ping печатает в stdout ровно один JSON и завершается с кодом 0 быстро.
case "${1:-}" in
  ping)
    echo '{"protocol_version":"1.0","supported_tasks":["T2"],"self_trained_rl":false,"preferred_speed_mps":null}'
    exit 0 ;;
esac
set -e
source /opt/ros/humble/setup.bash >&2
[ -f /home/ros/go1_ws/install/setup.bash ] && source /home/ros/go1_ws/install/setup.bash >&2
case "${1:-}" in
  T2) exec python3 /opt/t2/safety_block.py ;;
  *)  echo "Реализовано только T2, получено: ${1:-<пусто>}" >&2; exit 2 ;;
esac
