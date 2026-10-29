# Choosing the supplier email writer and validator

**2026-10-25.** How the email agent's model and effort, its prompt, and the email validator were
chosen.

## 1. What the agent does

The email agent writes only the requests paragraph of an email to the supplier; code adds the
greeting, the purpose, the item list, the closing and the signature. Its input is the supplier, the
items (name, quantity, requested delivery date), the analysis agent's questions marked for the
supplier, the Critical items (for the earliest-dispatch question) and the supplier's items outside
this order. It never receives stock figures, risk levels or internal questions.

What it must do: ask what the questions need, in the tone of a request to a business partner,
without revealing anything internal and without naming a problem the supplier had before. The email
leaves the company, so a past problem named in it tells the supplier what the company knows.

## 2. How it was tested

- The same 25 inputs for every setting: the email input of each of the 25 evaluation analyses
  (`scripts/compare_email_models.py`).
- Seven settings: `gpt-5.4-mini` at efforts none and low, `gpt-6-luna`
  none, low and medium, `gpt-6.1-sol` low and medium. No Claude model, since Haiku checks this text;
  no Gemini, unavailable in testing.
- Each email went through the production loop: draft, validator, a rewrite after a fail, at most
  three drafts.
- Checks E1–E7 (`docs/eval_rubric.md`): E3 in code, the rest by the judge (`claude-haiku-5-5`, high
  effort), three runs each, a check failed when two of the three runs failed it. Every final email
  was also read by hand, before the judge's verdicts were seen.
- Three rounds: the prompt as it was, then two revisions.

## 3. What the outputs showed

### The deciding difference: how a supplier's past problem is asked about

The analyst's questions often name what went wrong ("Was the loose-terminal failure in the January
PowerModuleZ batch isolated…?"). The same question, written into the email by each model:

| Setting | Wording in the email |
|---|---|
| `gpt-5.4-mini` none | "whether the loose-terminal issue in the January PowerModuleZ batch was isolated" |
| `gpt-6-luna` none | "whether the cause of the loose-terminal failure affects any products in this order" |
| `gpt-6.1-sol` low | "how terminal integrity will be verified for the items listed above" |
| `gpt-6.1-sol` medium | "whether any terminal-integrity considerations apply to them" |

- **`gpt-5.4-mini`** copies the past problem in, its month and a product outside the order included,
  whatever the prompt says. It once also wrote an Armenian word into an email.
- **`gpt-6-luna`** softens the problem words but keeps them ("the cause of the loose-terminal
  failure", "any product-standard discrepancies").
- **`gpt-6.1-sol`** turns them into ordinary questions about this order, revealing nothing the
  company has heard.

### Final round (round 3, both the prompt and the checks revised; 25 emails each)

| Setting | Past problem (E2) | Point missed (E4) | Question repeated (E5) | Judge: any fail | Reading: emails with an issue | Time | Cost per email |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gpt-5.4-mini` none | 6 | 0 | 3 | 6 | 7 | 2.4 s | $0.0010 |
| `gpt-5.4-mini` low | 2 | 1 | 4 | 6 | 6 | 2.5 s | $0.0013 |
| `gpt-6-luna` none | 2 | 0 | 1 | 3 | 3 | 2.3 s | $0.0001 |
| `gpt-6-luna` low | 2 | 0 | 0 | 2 | 1 | 3.4 s | $0.0002 |
| `gpt-6-luna` medium | 1 | 0 | 1 | 2 | 0 | 5.2 s | $0.0003 |
| **`gpt-6.1-sol` low** | 0 | 0 | 0 | **0** | **0** | **3.3 s** | $0.0024 |
| `gpt-6.1-sol` medium | 0 | 0 | 0 | 0 | 0 | 5.0 s | $0.0033 |

No setting revealed internal information (E1).

### What the rounds changed

- **The prompt had to apply its style to every question, not make an exception.** A rule to use a
  question "as it is where it reads well", placed before the rule on past problems, let every
  question through, since the analyst's points are questions. Applying the email's style to every
  point, with no past problem whatever a point says, fixed it for Sol and Luna, not for
  `gpt-5.4-mini`.
- **The checks were revised as well as the prompt.** The judge had failed questions the prompt
  requires to be left out, counted a required question as a repeat, and failed an indirect question
  for not naming the problem itself. Each check's wording was revised and the same emails scored
  again; after that the judge and the hand reading agreed.

## 4. The validator

The validator checks each draft for what must never reach the supplier (E1–E3) and sends it back to
be rewritten if it finds any.

**Why a new design.** A keyword check followed by `gpt-4o-mini`, run on the three rounds, passed
every email, "the loose-terminal issue in the January PowerModuleZ batch" included;
its only fails were false alarms. It looked only for what the email agent is never given (stock
figures, risk levels), and it was of the writer's family.

**What replaced it:** code checks first (internal terms; an item outside the order named by name or
code), then a model told to fail internal information, a past problem of the supplier or when it
happened, and an item outside the order referred to in other words.

**Test** (`scripts/compare_email_validators.py`): the 175 round-3 emails, 13 of them with a real
fault, and Sol low's 25 clean emails with one fault planted in each copy (an item outside the order
named or referred to, a past problem with its month, weeks of stock left), 239 emails in all.

| Validator | Real faults caught | Planted faults caught | Clean emails failed | Of Sol low's 25 | Time | Cost per check |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-4o-mini` | 13/13 | 61/64 | 55/162 (34%) | 10 | 0.8 s | $0.00005 |
| Haiku, low | 12/13 | 61/64 | 47/162 (29%) | 8 | 1.9 s | $0.00014 |
| **Haiku, medium** | **13/13** | 60/64 | **8/162 (4.9%)** | **0** | 2.2 s | $0.00018 |
| Haiku, high | 11/13 | 59/64 | 8/162 (4.9%) | 0 | 2.4 s | $0.00021 |

Every setting caught the planted past problems and internal figures; they differ in what they fail
wrongly. A false alarm sends a good email back to be rewritten. `gpt-4o-mini` failed a third of the
clean emails for reasons that were simply wrong (it took the item the email is about for one outside
it). Haiku at medium failed 8 of 162, none of Sol low's; seven of the eight were traces of a past
problem read more strictly than the judge's E2 ("expected recovery timetable and backlog clearance
schedule"). At high effort it missed two real faults for no fewer false alarms. Its one planted miss
more than `gpt-4o-mini` is an item referred to without its name in a deliberately vague wording ("the
other product we usually order from you"); a name or code is caught by code every time.

## 5. Decision

**Writer: `gpt-6.1-sol` at effort low.** Against `gpt-6-luna` medium, the other setting with no issue
found by reading:

| | `gpt-6.1-sol` low | `gpt-6-luna` medium |
|---|---|---|
| A supplier's past problem | asked as an ordinary question about this order | problem words kept, softened |
| Time per email | **3.3 s** | 5.2 s |
| Cost per email | $0.0024 | $0.0003 |

- **It asks indirectly where it matters.** The email goes outside the company; a softened trace of a
  past problem still tells the supplier what the company knows. This was the deciding difference.
- **It is faster,** the fastest of the settings without an issue.
- **It costs more,** about eight times as much per email, which is about $0.21 more per 100
  suppliers, since each supplier gets one email.
- It is the same model and effort as the purchase request.

**Validator: `claude-haiku-5-5` at effort medium.** It catches every real fault and leaves Sol's
emails alone, and it is of a different family from the writer; about 2.2 s and $0.0002 per check.

**Checked in production since 2026-10-29:** besides the validator's E1–E3, the judge also checks the
final email on E4–E7 (every point asked, no question repeated, the earliest dispatch asked for
Critical items, the tone of a request), as warnings in the email's note.
