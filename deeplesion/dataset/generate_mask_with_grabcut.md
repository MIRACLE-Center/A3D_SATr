# Data Process
generate mask from bounding box and recists through GrabCut [ref paper](https://www.microsoft.com/en-us/research/publication/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts/).
annotation file with mask will be generated in dataroot.
``` shell
python mask_generateor.py --dataroot ${dataroot path} --split ${split} --output_filename${output annotation pickle filename}
```


