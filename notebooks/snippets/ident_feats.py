# %%
from __future__ import annotations

import interpret
import interpret.glassbox
import mydatasets.aaco
import torch as th

# %%
ebc = interpret.glassbox.ExplainableBoostingClassifier()
tcube, vcube = mydatasets.aaco.load_aaco_data("cube_20_0.3", to_normalize=False)
ebc.fit(tcube["xs"].numpy(), tcube["ys"].numpy())

# %%
interpret.show(ebc.explain_global())

# %%
imp = th.as_tensor(ebc.term_importances(), dtype=th.float32)
print(imp.numpy())
print(th.argsort(imp, descending=True).numpy())

# %%
