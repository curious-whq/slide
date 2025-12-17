date
LITMUSOPTS="${@:-$LITMUSOPTS}"
SLEEP=0
if [ ! -f SB.no ]; then
cat <<'EOF'
%%%%%%%%%%%%%%%%%%%%%%%%%
% Results for SB.litmus %
%%%%%%%%%%%%%%%%%%%%%%%%%
RISCV SB
"PodWR Fre PodWR Fre"

{0:x5=1; 0:x6=x; 0:x8=y; 1:x5=1; 1:x6=y; 1:x8=x;}

 P0          | P1          ;
 sw x5,0(x6) | sw x5,0(x6) ;
 lw x7,0(x8) | lw x7,0(x8) ;

exists (0:x7=0 /\ 1:x7=0)
Generated assembler
EOF
cat SB.t
./SB.exe -q $LITMUSOPTS
fi
sleep $SLEEP

cat <<'EOF'
Revision exported, version 7.56
Command line: litmus7 -mach ./riscv.cfg -avail 4 SB.litmus -o SB
Parameters
#define SIZE_OF_TEST 2000
#define NUMBER_OF_RUN 20000
#define AVAIL 4
#define STRIDE (-1)
#define MAX_LOOP 0
/* gcc options: -Wall -std=gnu99 -O2 -m32 -pthread */
/* barrier: user */
/* launch: changing */
/* affinity: none */
/* alloc: dynamic */
/* memory: direct */
/* safer: write */
/* preload: random */
/* speedcheck: no */
/* proc used: 4 */
EOF
head -1 comp.sh
echo "LITMUSOPTS=$LITMUSOPTS"
date
