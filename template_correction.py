import pandas as pd
import re
import os

input_dir = './full_dataset/'
datasets = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS",
]


# Ground truth correction function from LUNAR
def correct_single_template(template, user_strings=None):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean) # we don't use this
    US (User String) # we don't use this
    DG (Digit)
    HEX (Hex Variables)
    PS (Path-like String) # we don't use this
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)

    """

    # boolean = {}
    # default_strings = {}
    path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    # if user_strings:
        # default_strings = default_strings.union(user_strings)

    # apply DS
    template = template.strip()
    template = re.sub(r'\s+', ' ', template)

    # apply PS
    # p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
    # new_p_tokens = []
    # for p_token in p_tokens:
        # if re.match(r'^(\/[^\/]+)+$', p_token):
            # p_token = '<*>'
        # new_p_tokens.append(p_token)
    # template = ''.join(new_p_tokens)

    # tokenize for the remaining rules
    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)  # tokenizing while keeping delimiters
    new_tokens = []
    for token in tokens:
        # apply BL, US
        # for to_replace in boolean.union(default_strings):
            # if token.lower() == to_replace.lower():
                # token = '<*>'

        # apply DG
        if re.match(r'^\d+$', token):
            token = '<*>'

        # newly added by me
        # apply Hex
        if re.match(r'0x[0-9a-fA-F]+', token):
            token = '<*>'
        while "0x<*>" in token:
            token = token.replace("0x<*>", "<*>")

        # apply WV
        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
            if token != '<*>/<*>':  # need to check this because `/` is not a deliminator
                token = '<*>'

        # collect the result
        new_tokens.append(token)

    # make the template using new_tokens
    template = ''.join(new_tokens)

    # Substitute consecutive variables only if separated with any delimiter including "." (DV)
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    # Substitute consecutive variables only if not separated with any delimiter including space (CV)
    # NOTE: this should be done at the end
    #print("CV: ", template)
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break
    #print("CV: ", template)

    while " #<*># " in template:
        template = template.replace(" #<*># ", " <*> ")

    while " #<*> " in template:
        template = template.replace(" #<*> ", " <*> ")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>#<*>" in template:
        template = template.replace("<*>#<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while "<*>@<*>" in template:
        template = template.replace("<*>@<*>", "<*>")

    while "<*>.<*>" in template:
        template = template.replace("<*>.<*>", "<*>")

    while ' "<*>" ' in template:
        template = template.replace(' "<*>" ', ' <*> ')

    while " '<*>' " in template:
        template = template.replace(" '<*>' ", " <*> ")

    while "<*><*>" in template:
        template = template.replace("<*><*>", "<*>")

    # newly added by me
    while " <*>. " in template:
        template = template.replace(" <*>. ", " <*> ")
    while " <*>, " in template:
        template = template.replace(" <*>, ", " <*> ")
    while "<*>+<*>" in template:
        template = template.replace("<*>+<*>", "<*>")
    while "<*>##<*>" in template:
        template = template.replace("<*>##<*>", "<*>")
    while "#<*>#" in template:
        template = template.replace("#<*>#", "<*>")
    while "<*>-<*>" in template:
        template = template.replace("<*>-<*>", "<*>")
    while " <*> <*> " in template:
        template = template.replace(" <*> <*> ", " <*> ")
    while template.endswith(" <*> <*>"):
        template = template[:-8] + " <*>"
    while template.startswith("<*> <*> "):
        template = "<*> " + template[8:]

    # newly added by me
    while "<*>,<*>" in template:
        template = template.replace("<*>,<*>", "<*>")
    while "(<*> <*>)" in template:
        template = template.replace("(<*> <*>)", "(<*>)")
    while " /<*> " in template:
        template = template.replace(" /<*> ", " <*> ")
    if template.endswith(" /<*>"):
        template = template[:-5] + " <*>"

    # Attribute key-value pair
    if template.count("=<*>") >= 3:
        template = template.replace("= ", "=<*> ")
    return template


# Correct the templates
for dataset in datasets:
    groundtruth_path = os.path.join(input_dir, f"{dataset}/{dataset}_full.log_structured.csv")
    groundtruth = pd.read_csv(groundtruth_path, dtype=str)
    groundtruth['EventTemplate'] = groundtruth['EventTemplate'].apply(lambda x: correct_single_template(x))
    groundtruth.to_csv(os.path.join(input_dir, f"{dataset}/{dataset}_full.log_structured_corrected.csv"), index=False)
    
    templates_path = os.path.join(input_dir, f"{dataset}/{dataset}_full.log_templates.csv")
    templates = pd.read_csv(templates_path, dtype=str)
    templates['EventTemplate'] = templates['EventTemplate'].apply(lambda x: correct_single_template(x))
    templates.to_csv(os.path.join(input_dir, f"{dataset}/{dataset}_full.log_templates_corrected.csv"), index=False)
