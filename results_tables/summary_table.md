# SafeLIBERO Benchmark Results — Summary

**Metrics:** TSR = Task Success Rate, CAR = Collision Avoidance Rate, ETS = Execution Time Steps (mean)

**Episodes:** 50 per task × 4 tasks = 200 episodes per condition


| Suite   | Level | OVL TSR | OVL CAR | OVL ETS | Pi0.5 TSR | Pi0.5 CAR | Pi0.5 ETS |
|---------|-------|---------|---------|---------|-----------|-----------|-----------|
| Spatial | I     | 42.0%   | 12.0%   | 219.2   | 67.0%     | 14.0%     | 188.2     |
| Spatial | II    | 35.0%   | 11.5%   | 235.6   | 55.5%     | 12.0%     | 208.1     |
| Object  | I     | 0.0%    | 1.5%    | 300.0   | 40.5%     | 14.0%     | 252.5     |
| Object  | II    | 42.5%   | 26.0%   | 241.4   | 74.0%     | 25.0%     | 192.8     |
| Goal    | I     | 18.0%   | 15.5%   | 262.1   | 51.0%     | 23.0%     | 168.5     |
| Goal    | II    | 21.0%   | 6.0%    | 255.4   | 66.5%     | 35.0%     | 172.1     |
| Long    | I     | 3.0%    | 15.0%   | 547.1   | 23.0%     | 60.5%     | 180.2     |
| Long    | II    | 16.0%   | 8.0%    | 506.1   | 36.5%     | 62.5%     | 199.6     |

> Note: Pi0.5 Long L-I tasks 2–3 show ETS=0 / CAR=100% — these tasks appear to have had 0 episodes execute (possible suite-level task availability issue).