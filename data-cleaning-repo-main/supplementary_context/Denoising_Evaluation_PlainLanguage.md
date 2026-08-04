# How We Verify Our Cleaned Storybook Data
### A plain-language guide for the general reader

This guide explains, without technical jargon, how we make sure the storybook files we prepare are properly cleaned. It is written for anyone who wants to understand the subject in general — no background in computers or coding is needed.

---

## The situation

Our project teaches a computer to judge how difficult a children's story is to read, across several Philippine languages. To do that, we collect real children's storybooks and feed the computer the text of each story.

The problem is that the original files are cluttered. Around each story sits a lot of *packaging*: a title page, the names of the author and illustrator, a copyright or license notice, page numbers, and often a set of quiz questions at the end. The computer should learn from the **story itself**, not from the packaging. So before anything else, we clean each file — we strip away the packaging and keep the story.

That raises the question this guide answers: **How do we know the cleaning was done correctly?**

The honest difficulty is that these stories are written in languages our team does not speak, and we do not have native speakers available to read and check all ~1,480 of them. So we cannot simply ask someone, "Does this cleaned story read correctly?"

---

## The key idea

Here is the insight that makes verification possible without speaking the language:

> **We never rewrite, translate, or reword anything. We only remove packaging.**

Because of that, we don't need to judge whether the cleaned story *reads well* — that would require a fluent speaker. We only need to confirm two things:

1. We removed **only** packaging (we didn't accidentally cut part of the story).
2. We kept the **whole** story (we didn't leave packaging behind).

Both of those are questions about the *shape* of the text, not its *meaning* — and you can answer them without understanding the words. Think about it: even in a language you can't read, you can usually tell a page number, a copyright line, or a list of names apart from an actual story.

---

## How we check — three simple checks

**1. We compare against a trusted answer key.**
For two of the languages, the original researchers who published the source material also published exact measurements of every story — how many words it has, how many sentences. We measure our own cleaned copies the same way and confirm we get the identical numbers.

*Think of it like:* testing a kitchen scale by weighing a known one-kilogram weight. If the scale reads exactly one kilogram, you can trust it. Matching the published answer key proves our measuring tools read and count the text correctly.

**2. We keep a receipt for everything we remove.**
For every single story, we keep the original file and the cleaned file side by side, and we produce an itemized list of exactly what was taken out — and where. Then we confirm each removed item is genuinely packaging, not story.

There's a clever shortcut here. Every language has small "glue" words that hold sentences together (the equivalents of *the, and, in, is*). These glue words appear constantly in real stories, but almost never in packaging like credits or copyright lines. So if one of those glue words ever goes missing from the *middle* of a story, that's a red flag that we may have cut real text — and we set that story aside for a person to look at.

*Think of it like:* an editor who can tell that a paragraph is missing from a page, even in a language they can't read, because the sentences no longer flow.

**3. We enforce a few safety rules.**
Some rules must hold for every story, no exceptions:
- **No letters may be lost.** These languages use special letters and accent marks (for example, ñ and accented vowels). Every one of them must survive the cleaning untouched.
- **Cleaning must be consistent.** Running the process again produces exactly the same result.
- **No leftover packaging.** No stray English copyright or publisher line may remain inside a story.

---

## What counts as "properly cleaned"

We hold every file to a clear standard. A story is accepted as properly cleaned when **all** of these are true:

- All the safety rules hold (no lost letters, consistent result, no leftover packaging).
- Where a trusted answer key exists, our measurements match it.
- The short list of flagged items has been reviewed by a person and confirmed to be packaging, not lost story.

Files that turn out to be nothing but packaging — for example, a standalone copyright page with no actual story — are set aside and recorded, rather than forced into the collection.

---

## What we end up with

Across the whole collection:

- **About 1,480 stories** in **seven languages**.
- The answer-key check **matches for every Cebuano story and all but two Bikol stories** — and those two differ by only two words each, for a well-understood harmless reason.
- **Every one of the ~1,480 stories keeps all its letters and accent marks.**
- **No leftover English packaging** anywhere in the stories.
- After all the automatic checks, only a **handful of items** needed a person's eyes — and each one turned out to be packaging (such as a story printed with its title in two languages), not lost story text.

To give a sense of what gets removed: the large majority is publisher and license boilerplate, followed by author biographies, then author and illustrator credits. The actual stories are kept intact.

---

## Why this is trustworthy

- **The checks are sensitive enough to catch real problems.** For instance, if a story's opening happens to sit right next to a publisher's notice, a careless clean-up could cut the whole story short — and these checks catch exactly that kind of slip and point to the specific story and spot. Nothing has to be caught by chance.
- **Everything is written down and repeatable.** Anyone can run the same checks and get the same result, and there is a complete, itemized record of what was removed from every story. Verification does not depend on trusting any one person's memory or eyesight.

---

## What this does and does not promise

It is worth being clear about the boundaries:

- **It does guarantee** that we removed only packaging, kept whole stories, and preserved every letter — and it provides a complete record proving it.
- **It does not claim** to judge the literary quality of a story or the accuracy of any translation. Because we never change the words, those questions simply don't arise.
- For two languages we have the extra reassurance of an outside answer key; for the others, the "receipt" and "safety-rule" checks do the job without needing one — helped by the fact that most of their packaging is in English, which is easy to spot inside a non-English story.

---

## A note on the languages

All seven languages receive the core checks — the receipts for what was removed and the safety rules. Two of them (Cebuano and Bikol) also have a published answer key, which gives an additional, independent confirmation. The other languages do not have such a key available, which is a normal limitation of working with under-served languages; the core checks do not depend on one.

---

## Where the records live

The process produces plain records that anyone can open and read:

- a **scorecard** summarizing whether each check passed,
- a short **review list** of the few items a person needs to glance at, and
- a complete, itemized **record of everything removed** from every story, each with its location.

Together, these turn "we cleaned the data" into "here is the evidence that the data is clean" — evidence that does not require reading the languages to trust.
