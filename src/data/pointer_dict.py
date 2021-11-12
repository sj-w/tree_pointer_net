import fairseq.data


class PointerDict(fairseq.data.Dictionary):
    def __init__(self, special_tokens=("pos", "neu", "neg"), pad='<pad>', eos='</s>', unk='<unk>'):
        super(PointerDict, self).__init__(pad, eos, unk)
        for t in special_tokens:
            self.add_symbol(f"<{t}>")

        self.added_special_start = self.nspecial
        self.added_special_end = self.added_special_start + len(special_tokens)