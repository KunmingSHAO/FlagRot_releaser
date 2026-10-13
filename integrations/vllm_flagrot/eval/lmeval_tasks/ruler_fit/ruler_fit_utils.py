"""lm-eval RULER tasks re-exported unchanged for the `*_fit` task variants (configs/ruler128k.yaml).

The `_fit` yamls are the lm-eval RULER yamls with one extra metric key (the 128k budget 130816 = 131072 - 256): lm-eval's
RULER sizes the raw input so that input + answer budget <= max_seq_length, without the chat template, so at 131072 a few
requests exceed Llama-3.1's native 131072-token window once the template is added (vLLM rejects them).  The 256 reserved
tokens hold the chat template; data generation, prompts and scoring are lm-eval's own code."""
from lm_eval.tasks.ruler.common_utils import aggregate_metrics, process_results, process_results_part  # noqa: F401
from lm_eval.tasks.ruler.cwe_utils import get_cw_dataset  # noqa: F401
from lm_eval.tasks.ruler.fwe_utils import fwe_download  # noqa: F401
from lm_eval.tasks.ruler.niah_utils import (niah_multikey_1, niah_multikey_2, niah_multikey_3,  # noqa: F401
                                            niah_multiquery, niah_multivalue, niah_single_1, niah_single_2,
                                            niah_single_3)
from lm_eval.tasks.ruler.qa_utils import get_hotpotqa, get_squad  # noqa: F401
from lm_eval.tasks.ruler.vt_utils import get_vt_dataset  # noqa: F401
