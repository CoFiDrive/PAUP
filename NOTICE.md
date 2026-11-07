# Attribution and external components

The MIT license in this repository applies to the PAUP-specific implementation
and training utilities distributed here. It does not relicense external models,
datasets, source packages or checkpoints.

PAUP was developed and evaluated on cooperative driving / predictive perception
infrastructure associated with the following projects:

- **Select2Drive**, distributed predictive perception and pragmatic communication:
  https://github.com/ZJUNICE/Select2Drive
  Paper: https://arxiv.org/abs/2501.12040
- **V2Xverse / CoDriving**, collaborative-driving datasets and backbone planning:
  https://github.com/CollaborativePerception/V2Xverse
  Paper: https://arxiv.org/abs/2404.09496
- **OpenCOOD**, cooperative perception infrastructure:
  https://github.com/DerrickXuNu/OpenCOOD
- **OpenSTL / DMVFN**, predictive heatmap backbone infrastructure:
  https://github.com/chengtan9907/OpenSTL

Their original authors retain their copyrights. Some external source files
and the OpenCOOD academic license restrict redistribution. Accordingly, this
repository does not vendor those backend implementations or trained weights.
Install any required backend independently under its own applicable terms.
The backend adapter calls its public Python interfaces and does not erase
or replace upstream copyright notices.

The spatiotemporal planner in this repository is a native-PyTorch reference
implementation of the architecture used by PAUP, with a checkpoint-name
conversion for privately supplied compatible planner states. The architectural
relationship to the preceding cooperative-driving work should be cited when
using it. The PAUP operators do not add learned parameters to the perception
backbone.
