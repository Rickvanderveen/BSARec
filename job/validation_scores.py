import os
import re
import argparse

home = os.environ.get("HOME")
# folder containing hyperparameter search results
outputs_path = os.path.join(home, "BSARec/job/out/hparams")

# paths
outfile = lambda x : os.path.join(outputs_path, x)
hp_txt = lambda x : os.path.join(home, f"BSARec/job/{x}_hparams.txt")

# hyperparams txts
hparams_txt = {
    "Bert4Rec" : hp_txt("bert4rec"), "DuoRec" : hp_txt("duorec"), "FEARec" : hp_txt("fearec"),
    "SASRec" : hp_txt("sasrec"), "BSARec" : hp_txt("bsarec")
}

def validation_scores(model, dataset):
    # out files
    outputs = sorted([outfile(f) for f in os.listdir(outputs_path) if model in f and dataset in f])
    hparams = open(hparams_txt[model]).read().splitlines()

    results = []
    for hp, out in zip(hparams, outputs):
        hs = hp.split()
        # hyper parameters
        cur_dict = {hs[i*2][2:]: hs[i*2+1] for i in range(len(hs)//2)}

        # last validation scores of outfile
        val = eval(re.findall(r"\{'Epoch'.*\}", open(out).read())[-1])
        for k,v in val.items():
            cur_dict[k] = float(v) 

        results.append(cur_dict)

    return results

if __name__ == "__main__":

    # should match the name of the hyperparameter search outfile
    models = ["Bert4Rec", "DuoRec", "FEARec", "SASRec", "BSARec"]
    datasets = ["Diginetica", "LastFM"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help=f"model from {models}")
    parser.add_argument("--dataset", required=True, help=f"dataset from {datasets}")
    args = parser.parse_args()

    if args.model not in models:
        raise Exception(f"model not in {models}")
    if args.dataset not in datasets:
        raise Exception(f"dataset not in {datasets}")

    res = validation_scores(args.model, args.dataset)
    print(*res, sep="\n")