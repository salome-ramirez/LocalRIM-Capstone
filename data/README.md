# Data

The documents are from the Philip Morris Records collection of the [UCSF Industry Documents Library](https://www.industrydocuments.ucsf.edu/), which makes them publicly available. See the library's [Copyright and Fair Use](https://www.industrydocuments.ucsf.edu/about-idl/copyright/) page for the terms of use. This repository is a research project and is not affiliated with or endorsed by the library.

The PDFs and page images are not included. `pipeline/01_sample/download_pdfs.py` downloads the PDFs from the library and renders the first pages.

What is included, in `data/pm_sample/`:

| File or folder | Contents |
|---|---|
| `manifest.csv` | the 1,403 sampled documents with the library's catalogue fields (title, date, type, pages, filing area, Bates number, collection) |
| `metadata_full.csv` | every catalogue field the library holds for each document |
| `pdf_info.csv` | page count, scan resolution and size of each PDF |
| `page_profile.csv` | scan-quality measures and group of each first page |
| `ocr/`, `ocr_tess_native/`, `ocr_paddle/`, `ocr_vlm/`, `ocr_ucsf/` | the text of each first page as read by OCR engines A to E (and word boxes where the engine gives them) |
| `ocr_text/`, `ocr_text_clean/` | the classifier's text options, before and after removing each page's own Bates number |

Five documents (smmw0220, nkjv0160, tqyl0166, tzmx0220, kpvf0220) were flagged during labeling as containing personal data. Their text, images, titles and notes are not included; only their IDs, labels, scores and embedding vectors remain, which the published numbers need.

If you are a rights holder or are named in one of these documents and want something removed, please open an issue in this repository and it will be taken down.
