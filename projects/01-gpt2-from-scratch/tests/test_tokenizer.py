import pytest

from gpt2_scratch.tokenizer import (
    BPETokenizer,
    CharTokenizer,
    load_tokenizer,
    merge_pair,
    save_tokenizer,
)


def test_character_mapping_is_deterministic():
    tokenizer = CharTokenizer("cabca\n")
    assert tokenizer.encode("abc") == [1, 2, 3]
    assert tokenizer.decode(tokenizer.encode("cab\n")) == "cab\n"
    assert tokenizer.to_dict() == CharTokenizer("\nabc").to_dict()


def test_character_unknown_is_explicit():
    with pytest.raises(ValueError, match="unknown character"):
        CharTokenizer("abc").encode("z")


def test_character_negative_id_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        CharTokenizer("abc").decode([-1])


@pytest.mark.parametrize("text", ["", "hello\nworld", "Bonjour, café!", "你好 🌍", "  a\t\nb"])
def test_bpe_roundtrips_unseen_unicode(text):
    tokenizer = BPETokenizer.fit("banana banana banana", 270)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_learns_real_merges_with_deterministic_ties():
    tokenizer = BPETokenizer.fit("banana banana", 257)
    assert tokenizer.merges == [(ord("a"), ord("n"))]
    assert tokenizer.encode("banana") == [ord("b"), 256, 256, ord("a")]
    assert len(tokenizer.encode("banana")) < len("banana")


def test_merges_do_not_overlap():
    assert merge_pair((1, 1, 1), (1, 1), 256) == (256, 1)


@pytest.mark.parametrize("kind", ["char", "bpe"])
def test_tokenizer_save_load_preserves_ids(tmp_path, kind):
    tokenizer = CharTokenizer("banana\n") if kind == "char" else BPETokenizer.fit("banana\n" * 10)
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tokenizer, path)
    loaded = load_tokenizer(path)
    assert loaded.to_dict() == tokenizer.to_dict()
    assert loaded.encode("banana\n") == tokenizer.encode("banana\n")


def test_byte_tokenizer_rejects_bad_merge_references():
    with pytest.raises(ValueError, match="earlier"):
        BPETokenizer([(256, 3)])
