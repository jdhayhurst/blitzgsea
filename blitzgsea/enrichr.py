import itertools
import json
import os
import re
import shutil
import ssl
import urllib.request


def get_library(library: str) -> dict[str, list[str]]:
    """Load gene set library from Enrichr as a dictionary."""
    return read_gmt(load_library(library))


def list_libraries() -> list:
    print(get_config())
    return load_json(get_config()["LIBRARY_LIST_URL"])["library"]


def load_library(library: str, overwrite: bool = False, verbose: bool = False) -> str:
    if not os.path.exists(get_data_path() + library or overwrite):
        if verbose:
            print("Download Enrichr geneset library")
        urlretrieve(get_config()["LIBRARY_DOWNLOAD_URL"] + library, get_data_path() + library)
    else:
        if verbose:
            print(f'File cached. To reload use load_library("{library}", overwrite=True) instead.')
    lib = read_gmt(get_data_path() + library)
    if verbose:
        print(f"# genesets: {len(lib)}")
    return get_data_path() + library


def print_libraries() -> None:
    """Print all available Enrichr gene set library names."""
    libs = list_libraries()
    for i, name in enumerate(libs):
        print(f"{i} - {name}")


def read_gmt(
    gmt_file: str,
    background_genes: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, list[str]]:
    with open(gmt_file) as file:
        lines = file.readlines()

    library: dict[str, list[str]] = {}
    background_set: set[str] = set()

    if background_genes and len(background_genes) > 1:
        background_set = {x.upper() for x in background_genes}

    for line in lines:
        sp = line.strip().split("\t")
        sp2 = [re.sub(",.*", "", value) for value in sp[2:]]
        sp2 = [x.upper() for x in sp2 if x]
        if background_set and len(background_set) > 2:
            geneset = list(set(sp2).intersection(background_set))
            if geneset:
                library[sp[0]] = geneset
        else:
            if sp2:
                library[sp[0]] = sp2

    ugenes = list(set(itertools.chain.from_iterable(library.values())))
    if verbose:
        print(
            f"Library loaded. Library contains {len(library)} gene sets. "
            f"{len(ugenes)} unique genes found."
        )
    return library


def load_json(url: str) -> dict:
    context = ssl._create_unverified_context()
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=context) as fr:
        r = fr.read()
    return json.loads(r.decode("utf-8"))


def get_config() -> dict:
    config_url = os.path.join(os.path.dirname(__file__), "data/config.json")
    with open(config_url) as json_file:
        return json.load(json_file)


def get_data_path() -> str:
    return os.path.join(os.path.dirname(__file__), "data/")


def urlretrieve(req: str, filename: str) -> None:
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(req, context=context) as fr:
        with open(filename, "wb") as fw:
            shutil.copyfileobj(fr, fw)
