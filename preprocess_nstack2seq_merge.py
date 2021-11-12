import os
import shutil

from fairseq import options, tasks
from fairseq.binarizer import Binarizer
from fairseq.data import dictionary
from fairseq.data import indexed_dataset
from fairseq.tokenizer import tokenize_line
from fairseq.utils import import_user_module

import src
from src.data.pointer_dict import PointerDict

CoreNLPTreeBuilder = src.dptree.CoreNLPTreeBuilder

BinarizerDataset = src.binarization.BinarizerDataset

NstackTreeBuilder = src.dptree.NstackTreeBuilder
NStackDataset = src.dptree.NStackDataset
NstackTreeTokenizer = src.nstack_tokenizer.NstackTreeTokenizer
NSTACK_KEYS = src.nstack_tokenizer.NSTACK_KEYS

NstackTreeMergeBinarizerDataset = src.binarization.NstackTreeMergeBinarizerDataset
NstackSeparateIndexedDatasetBuilder = src.data.NstackSeparateIndexedDatasetBuilder


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    ds = indexed_dataset.IndexedDatasetBuilder(
        dataset_dest_file(args, output_prefix, lang, "bin")
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos,
                             offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, lang):
    base = f"{args.destdir}/{output_prefix}"
    lang_part = (
        f".{args.source_lang}-{args.target_lang}.{lang}" if lang is not None else ""
    )
    return f"{base}{lang_part}"


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return f"{base}.{extension}"


def dataset_dest_prefix_dptree(args, output_prefix, lang, modality):
    assert lang is not None
    base = f"{args.destdir}/{output_prefix}"
    lang_part = (
        f".{args.source_lang}-{args.target_lang}.{lang}.{modality}" if lang is not None else ""
    )
    return f"{base}{lang_part}"


def dataset_dest_file_dptree(args, output_prefix, lang, extension, modality):
    base = dataset_dest_prefix_dptree(args, output_prefix, lang, modality)
    return f"{base}.{extension}"


def main(args):
    import_user_module(args)

    print(args)

    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source

    task = tasks.get_task(args.task)

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    if args.convert_raw:
        print(f'start --- args.convert_raw')
        raise NotImplementedError

    if args.convert_raw_only:
        print(f'Finish!.')
        return

    remove_root = not args.no_remove_root
    take_pos_tag = not args.no_take_pos_tag
    take_nodes = not args.no_take_nodes
    reverse_node = not args.no_reverse_node
    no_collapse = args.no_collapse
    # remove_root =, take_pos_tag =, take_nodes =
    print(f'remove_root: {remove_root}')
    print(f'take_pos_tag: {take_pos_tag}')
    print(f'take_nodes: {take_nodes}')
    print(f'reverse_node: {reverse_node}')
    print(f'no_collapse: {no_collapse}')

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_shared_nstack2seq_dictionary(_src_file, _tgt_file, _src_tst_file, _tgt_tst_file):
        d = dictionary.Dictionary()
        print(f'Build dict on src_file: {_src_file}')
        NstackTreeTokenizer.acquire_vocab_multithread(
            _src_file, d, tokenize_line, num_workers=args.workers,
            remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes,
            no_collapse=no_collapse,
        )
        NstackTreeTokenizer.acquire_vocab_multithread(
            _src_tst_file, d, tokenize_line, num_workers=args.workers,
            remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes,
            no_collapse=no_collapse,
        )
        print(f'Build dict on tgt_file: {_tgt_file}')
        dictionary.Dictionary.add_file_to_dictionary(_tgt_file, d, tokenize_line, num_workers=args.workers)
        dictionary.Dictionary.add_file_to_dictionary(_tgt_tst_file, d, tokenize_line, num_workers=args.workers)
        d.finalize(
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor
        )
        print(f'Finish building vocabulary: size {len(d)}')
        return d

    def build_nstack_source_dictionary(_src_file):
        d = dictionary.Dictionary()
        print(f'Build dict on src_file: {_src_file}')
        NstackTreeTokenizer.acquire_vocab_multithread(
            _src_file, d, tokenize_line, num_workers=args.workers,
            remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes,
            no_collapse=no_collapse,
        )
        d.finalize(
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor
        )
        print(f'Finish building src vocabulary: size {len(d)}')
        return d

    def build_target_dictionary(_tgt_file):
        # assert src ^ tgt
        print(f'Build dict on tgt: {_tgt_file}')
        d = task.build_dictionary(
            [_tgt_file],
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )
        print(f'Finish building tgt vocabulary: size {len(d)}')
        return d

    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))

    if args.joined_dictionary:
        assert not args.srcdict or not args.tgtdict, \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_file = f'{args.trainpref}.{args.source_lang}'
            tgt_file = f'{args.trainpref}.{args.target_lang}'
            src_tst_file = f'{args.testpref}.{args.source_lang}'
            tgt_tst_file = f'{args.testpref}.{args.target_lang}'
            src_dict = build_shared_nstack2seq_dictionary(src_file, tgt_file, src_tst_file, tgt_tst_file)
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_nstack_source_dictionary(train_path(args.source_lang))

        if target:
            if args.tgtdict:
                tgt_dict = task.load_dictionary(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_target_dictionary(train_path(args.target_lang))
        else:
            tgt_dict = None

    src_dict.save(dict_path(args.source_lang))
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args.target_lang))

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        print("| [{}] Dictionary: {} types".format(lang, len(vocab) - 1))

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        pool = None

        ds = indexed_dataset.IndexedDatasetBuilder(
            dataset_dest_file(args, output_prefix, lang, "bin")
        )

        def consumer(tensor):
            ds.add_item(tensor)

        stat = BinarizerDataset.export_binarized_dataset(
            input_file, vocab, consumer, add_if_not_exist=False, num_workers=num_workers,
        )

        ntok = stat['ntok']
        nseq = stat['nseq']
        nunk = stat['nunk']

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        print(
            "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                nseq,
                ntok,
                100 * nunk / ntok,
                vocab.unk_word,
            )
        )

    def make_binary_nstack_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        print("| [{}] Dictionary: {} types".format(lang, len(vocab) - 1))

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )

        dss = {
            modality: NstackSeparateIndexedDatasetBuilder(
                dataset_dest_file_dptree(args, output_prefix, lang, 'bin', modality))
            for modality in NSTACK_KEYS
        }

        def consumer(example):
            for modality, tensor in example.items():
                dss[modality].add_item(tensor)

        stat = NstackTreeMergeBinarizerDataset.export_binarized_separate_dataset(
            input_file, vocab, consumer, add_if_not_exist=False, num_workers=num_workers,
            remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes, reverse_node=reverse_node,
            no_collapse=no_collapse,
        )
        ntok = stat['ntok']
        nseq = stat['nseq']
        nunk = stat['nunk']

        for modality, ds in dss.items():
            ds.finalize(dataset_dest_file_dptree(args, output_prefix, lang, "idx", modality))

        print(
            "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                nseq,
                ntok,
                100 * nunk / ntok,
                vocab.unk_word,
            )
        )
        for modality, ds in dss.items():
            print(f'\t{modality}')

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args.output_format == "binary":
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)
        elif args.output_format == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)

    def make_dptree_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args.output_format != "binary":
            raise NotImplementedError(f'output format {args.output_format} not impl')

        make_binary_nstack_dataset(vocab, input_prefix, output_prefix, lang, num_workers)

    def make_all_tgt(lang, vocab):
        if args.trainpref:
            # print(f'!!!! Warning..... Not during en-fr target because already done!.....')
            make_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers)

        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix, lang, num_workers=args.eval_workers)
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args.eval_workers)

    def make_all_src(lang, vocab):
        if args.trainpref:
            # print(f'!!!! Warning..... Not during en-fr source because already done!.....')
            make_dptree_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers)

        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dptree_dataset(vocab, validpref, outprefix, lang, num_workers=args.eval_workers)

        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dptree_dataset(vocab, testpref, outprefix, lang, num_workers=args.eval_workers)

    make_all_src(args.source_lang, src_dict)
    if target:
        make_all_tgt(args.target_lang, tgt_dict)
        # print(f'No makign target')

    print("| Wrote preprocessed data to {}".format(args.destdir))

    if args.alignfile:
        raise NotImplementedError('alignfile Not impl at the moment')


def cli_main():
    parser = options.get_preprocessing_parser()
    # parser.add_argument("--output-format", metavar="FORMAT", default="binary",
    #                     choices=["binary", "raw"],
    #                     help="output format (optional)")
    group = parser.add_argument_group('Preprocessing')

    group.add_argument("--convert_raw", action="store_true", help="convert_raw")
    group.add_argument("--convert_raw_only", action="store_true", help="convert_raw")
    group.add_argument("--convert_with_bpe", action="store_true", help="convert_with_bpe")
    # group.add_argument("--bpe_code", action="store_true", help="convert_with_bpe")
    group.add_argument('--bpe_code', metavar='FILE', help='bpe_code')

    group.add_argument("--no_remove_root", action="store_true", help="no_remove_root")
    group.add_argument("--no_take_pos_tag", action="store_true", help="no_take_pos_tag")
    group.add_argument("--no_take_nodes", action="store_true", help="no_take_nodes")
    group.add_argument("--no_reverse_node", action="store_true", help="no_reverse_node")
    group.add_argument("--no_collapse", action="store_true", help="no_collapse")

    group.add_argument("--raw_workers", metavar="N", default=0, type=int, help="number of parallel workers")
    group.add_argument("--eval_workers", metavar="N", default=0, type=int, help="number of parallel workers")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
