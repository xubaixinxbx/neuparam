pwd
baseid=(3)
for item in ${baseid[@]}
do
echo $item
python training/exp_runner.py  \
--nepoch 2000 \
--conf ./confs/omni/omni.conf \
--scan_id $item \
--gpu 0 --expname _wlap
done