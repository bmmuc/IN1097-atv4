#!/bin/bash
# Guarda termica: registra Tctl a cada 5 s; se >= 88C pausa (SIGSTOP) os processos
# do experimento ate cair abaixo de 80C (SIGCONT). Pausas ficam registradas em
# results_automl/logs/thermal.log para identificar execucoes afetadas.
LOG=$(dirname "$0")/../results_automl/logs/thermal.log
PAT='hs_automl_|autogluon|ray::|raylet|gcs_server|loky'
paused=0; max=0
while true; do
  t=$(sensors 2>/dev/null | awk '/Tctl/ {gsub(/[+°C]/,"",$2); print int($2); exit}')
  [ -z "$t" ] && t=0
  [ "$t" -gt "$max" ] && max=$t
  now=$(date +%FT%T)
  if [ "$paused" -eq 0 ] && [ "$t" -ge 89 ]; then
    pgrep -f "$PAT" | xargs -r kill -STOP; paused=1; echo "$now PAUSE t=$t" >> $LOG
  elif [ "$paused" -eq 1 ] && [ "$t" -lt 80 ]; then
    pgrep -f "$PAT" | xargs -r kill -CONT; paused=0; echo "$now RESUME t=$t" >> $LOG
  fi
  [ $(( $(date +%s) % 300 )) -lt 5 ] && echo "$now t=$t max=$max paused=$paused" >> $LOG
  sleep 2
done
