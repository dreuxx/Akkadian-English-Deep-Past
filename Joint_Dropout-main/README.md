# **Joint Dropout for Machine Translation**

This project implements a **Joint Dropout** technique to improve generalization and robustness in **Neural Machine Translation (NMT)**. It includes scripts for preparing aligned data using the **eflomal** tool and processing it to generate training-ready files for the Joint Dropout pipeline.

---

## **Install Dependencies**

### **Python Libraries**
Install the required Python libraries:
```bash
pip install eflomal sacremoses subword-nmt nltk
```





### **Command-Line Tools**
Install atools for symmetrization:
```bash
sudo apt install atools
```



## **📋 Workflow**
### **Step 1: Generate Alignments**
Run the align_corpus.py script to generate forward and reverse alignments:
```bash
python src/align_corpus.py
```
This produces:

- data/output/de-en.fwd (forward alignment).
- data/output/de-en.rev (reverse alignment).


### **Step 2: Symmetrize Alignments**
Run the symmetrize.sh script to create a symmetrized alignment file:
```bash
bash src/symmetrize.sh
```

This produces:
- data/output/de-en.sym (symmetrized alignment file).



### **Step 3: Generate Priors (Optional but Recommended)**
Run the generate_priors.py script to create priors based on the alignments:
```bash
python src/generate_priors.py
```
This produces:
- data/output/de-en.priors (priors file to improve alignment accuracy).


### **Step 4: Merge Source and Target Files (Optional)**
If needed for additional workflows, merge the source and target files:
```bash
python src/merge_files.py
```
This produces:
- data/output/de-en.merged (merged file with ||| separator).

### **Step 5: Run Joint Dropout**
Once the input files are ready, configure the config.json file and run joint_dropout.py:
```bash
python src/joint_dropout.py
```

## **📤 Outputs**

Running joint_dropout.py will produce:

- Modified Source Sentences: data/output/output_src.txt
- Modified Target Sentences: data/output/output_trg.txt
These files are ready for use in downstream tasks, such as training a translation model.

## **🚨 Troubleshooting**

- Missing Tools: Ensure eflomal and atools are installed and available in your system path.
- Incorrect File Paths: Verify all file paths in config.json and scripts.
- Alignment Issues: If alignments are poor, consider preprocessing the data (e.g., tokenization, cleaning) or using priors.

## **📚 Acknowledgments**

This project uses:

- [eflomal](https://github.com/robertostling/eflomal): Efficient low-memory word alignment tool.
- [atools](https://github.com/clab/fast_align): Symmetrization utility from the fast_align toolkit.


## **📝 Citation**

If you use this project in your research, please cite the following paper:

**[Joint Dropout: Improving Generalizability in Low-Resource Neural Machine Translation through Phrase Pair Variables](https://aclanthology.org/2023.mtsummit-research.2/)**

### **BibTeX**
```bibtex
@inproceedings{araabi-etal-2023-joint,
    title = "Joint Dropout: Improving Generalizability in Low-Resource Neural Machine Translation through Phrase Pair Variables",
    author = "Araabi, Ali  and
      Niculae, Vlad  and
      Monz, Christof",
    editor = "Utiyama, Masao  and
      Wang, Rui",
    booktitle = "Proceedings of Machine Translation Summit XIX, Vol. 1: Research Track",
    month = sep,
    year = "2023",
    address = "Macau SAR, China",
    publisher = "Asia-Pacific Association for Machine Translation",
    url = "https://aclanthology.org/2023.mtsummit-research.2",
    pages = "12--25",
}

