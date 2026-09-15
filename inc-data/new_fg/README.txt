new_fg — Figure 5 (f)+(g) twin plots + slot data (v306 seed remine)

figures/
  Figure5_dblp_fg.{png,pdf}      DBLP 4-table UNION pure add / delete
  Figure5_ncvoter_fg.{png,pdf}   NCVoter pure add / delete

data/
  dblp_f/add{1,5,15,20,30}/results.csv
  dblp_g/del{1,5,15,20,30}/results.csv
  ncvoter_f/add{...}/results.csv
  ncvoter_g/del{...}/results.csv
  (+ SEED_OVERRIDE.txt, parameters.txt per slot)

seeds/
  SEED_OVERRIDES from Batch soft-monotone probe

Image: harbor.grandhoo.com/rock/piod/incminer-exp:v306
