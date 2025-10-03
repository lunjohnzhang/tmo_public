TO_UPLOAD="$1"

scp "-oProxyJump=xxq896@wallace.cs.bham.ac.uk" -r $TO_UPLOAD  sxl1242@bluebear.bham.ac.uk:/rds/projects/l/lehrepk-shishen/