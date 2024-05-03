import os
import sys
import ipdb
import pandas as pd
from gemba.cache import Cache
from gemba.gpt_api import GptApi
from gemba.CREDENTIALS import credentials
from gemba.gemba_mqm_utils import (
    TEMPLATE_GEMBA_MQM,
    apply_template,
    parse_mqm_answer,
    overall_quality_score_from_apt,
)

from absl import app, flags
from colorama import Fore


flags.DEFINE_string("source", None, "Filepath to the source file.")
flags.DEFINE_string("hypothesis", None, "Filepath to the translation file.")
flags.DEFINE_string("source_lang", None, "Source language name.")
flags.DEFINE_string("target_lang", None, "Target language name.")
flags.DEFINE_bool("verbose", False, "Verbose mode.")


model = "gpt-4-turbo-2024-04-09"

THRESHOLD = 85  # percent accuracy for pass


def main(argv):
    FLAGS = flags.FLAGS
    assert FLAGS.source is not None, "Source file must be provided."
    assert FLAGS.hypothesis is not None, "Hypothesis file must be provided."

    # check that source and hypothesis files exists
    if not os.path.isfile(FLAGS.source):
        print(f"Source file {FLAGS.source} does not exist.")
        sys.exit(1)
    if not os.path.isfile(FLAGS.hypothesis):
        print(f"Hypothesis file {FLAGS.hypothesis} does not exist.")
        sys.exit(1)

    assert FLAGS.source_lang is not None, "Source language name must be provided."
    assert FLAGS.target_lang is not None, "Target language name must be provided."

    # load both files and strip them
    with open(FLAGS.source, "r") as f:
        source = f.readlines()
    source = [x.strip() for x in source]
    with open(FLAGS.hypothesis, "r") as f:
        hypothesis = f.readlines()
    hypothesis = [x.strip() for x in hypothesis]

    assert len(source) == len(
        hypothesis
    ), "Source and hypothesis files must have the same number of lines."

    df = pd.DataFrame({"source_seg": source, "target_seg": hypothesis})
    df["source_lang"] = FLAGS.source_lang
    df["target_lang"] = FLAGS.target_lang

    df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)

    gptapi = GptApi(credentials, verbose=FLAGS.verbose)
    cache = Cache(f"{model}_GEMBA-MQM.jsonl")

    answers = gptapi.bulk_request(
        df,
        model,
        lambda x: parse_mqm_answer(x, list_mqm_errors=False, full_desc=True),
        cache=cache,
        max_tokens=500,
    )
    apt, ewc = 0, len(" ".join(source))
    failed = False
    for answer in answers:
        score = answer["answer"]
        apt += score
        if score == -25:
            failed = True

    print(Fore.CYAN + "Results")
    overall_quality_score = overall_quality_score_from_apt(-apt, ewc)
    print(f"Overall Quality Score: {overall_quality_score:.2f}")
    if failed:
        print(Fore.RED + "FAIL - Critical error in translation")
    elif overall_quality_score < THRESHOLD:
        print(Fore.RED + "FAIL")
    else:
        print(Fore.GREEN + "PASS")


if __name__ == "__main__":
    app.run(main)
