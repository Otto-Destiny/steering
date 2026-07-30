# Source Resolvers

A resolver recognizes, canonicalizes, and captures one source family. It returns source material and provenance;
it does not write approved graph knowledge directly.

## Resolution order

1. Normalize and validate the submitted URL or file.
2. Use the least privileged public representation.
3. Follow one author-attached primary link only after network-safety checks.
4. Use explicitly authorized browser capture only when public extraction is insufficient.
5. Preserve honest partial metadata when full text is unavailable.

For papers, prefer scholarly HTML or XML, then extracted PDF text, tables, and captions, then multimodal PDF
processing when ordinary extraction is inadequate. Initial coverage includes arXiv, OpenReview, ACL Anthology,
PubMed/PMC, DOI-linked publishers, proceedings, institutional repositories, and direct PDFs.

## X (Twitter)

X wraps **every** outbound link in a `t.co` shortlink. A capture that keeps the shortlink is not useful: no
resolver can rank or follow it, so the paper or repository the post was written to point at is silently lost.
Preserving the author's real destinations is therefore the defining requirement of this resolver, not a detail.

Public capture prefers the syndication payload that X's own embed widget requests. It carries the post body,
`entities.urls[].expanded_url` destinations, media with author-supplied alt text, and the publication timestamp.
The oEmbed blockquote is the fallback; it truncates the body and returns only shortlinks, so that path unwraps
them through the guarded fetcher before returning and marks the capture degraded.

Known limits, all reported rather than hidden:

- **Long-form posts.** Posts carrying a `note_tweet` are truncated to their opening by every public
  representation. These are marked `partial`, record `long_form_truncated`, and advise signed-in capture.
  No feature flag on the public endpoint returns the full body.
- **Withdrawn posts.** A `TweetTombstone` is authoritative. It fails with X's own reason and does not fall
  back, because a weaker path cannot contradict it and would replace a precise reason with a vague one.
- **Non-post URLs.** Profiles, search, and lists are refused with an actionable message. They must never reach
  the generic web resolver, which would capture X's login wall and store it as knowledge.

Every URL shape for one post — `?s=` share parameters, `/photo/1`, `/i/web/status/`, `statuses/`, `twitter.com`,
`mobile.`/`m.` hosts, and mixed-case handles — reduces to a single canonical `https://x.com/<handle>/status/<id>`.
Signed-in capture reuses the handle it reads from the thread so it deduplicates against the public capture of the
same post and can upgrade it once.

Signed-in capture is for what public capture genuinely cannot reach: the author's self-reply thread. It never
waits for a human to sign in inside a request; a missing session fails fast and points at the sign-in action.

### Recovering long-form bodies

X truncates long-form posts in every representation it publishes and exposes the remainder nowhere: neither
`note_tweet`, nor any widget feature flag, nor oEmbed returns the rest. FixTweet, the open-source service behind
`fxtwitter.com`, does return the complete body, so it is called **only** to recover text X withheld — never as
the primary capture, never for a post X already returned in full, and never for links or metadata. If it fails
or returns something no longer than what X gave, the capture keeps the honest truncated body and its advice,
because trading a declared limit for a silent one is the worse outcome. Pass `recover_long_form=False` to
`XResolver` to keep every request on X's own hosts.

### Reading the self-reply thread

Threads are the case that matters most in practice: authors routinely keep the paper or repository link *out* of
the root post to drive engagement into the replies, and no public X representation exposes replies at all. Only
an authorized signed-in capture can read them, and only the root author's own replies are kept — other people's
comments are discarded, and self-replies more than 48 hours from the root post fall outside the thread window.

Pasting a root-post link is enough. Ingestion applies one of three policies:

| Policy | Behaviour |
| --- | --- |
| `auto` (default) | Capture publicly first. Escalate to the signed-in browser **only** when the post actually has replies *and* a sign-in session already exists. |
| `always` | Always read the thread, and stop rather than quietly settle for a root-post-only capture. |
| `never` | Never open the browser. |

Under `auto` both preconditions are cheap: the reply count comes free with the public capture, and the session
check is a filesystem probe, so a user who never signed in pays nothing for the feature existing. If escalation
is attempted and fails, the public capture is kept and the artifact records `thread_escalation` with the reason —
degrading is acceptable, degrading invisibly is not.

Whatever the thread yields, links from **all** retained posts are pooled, unwrapped, and ranked together, and up
to `MAX_PRIMARY_SOURCES` of them are followed. A thread citing a paper and its repository keeps both, and they
still share a single extraction call.

### Capture surfaces, in priority order

Four ways in, ordered so the cheapest that can do the job wins:

| Surface | Cost | Reads replies | Needs |
| --- | --- | --- | --- |
| Public capture | free | no | nothing |
| Signed-in browser | free | yes | one interactive sign-in |
| X API | billed per read | yes | `STEERING_X_API_CLIENT_ID` + authorization |
| Bookmarklet import | free | n/a — supplies URLs | a click on your bookmarks page |

Public capture always runs first. Thread readers are then tried in order: the free browser before the
billed API, so paying for a read never pre-empts a session that already works. A reader that reports
itself unavailable costs nothing, which is why an unconfigured API is inert rather than a failure.

### When replies are read

Reading a thread costs about twelve seconds of browser time, or a billed API call. `STEERING_READ_THREADS`
decides when that is worth spending:

| Value | Behaviour |
| --- | --- |
| `when-needed` (default) | Only when the root post has no source worth following, or says the link is in the replies. |
| `always` | Whenever a post has replies. |
| `never` | Keep every capture to the root post. |

Under `when-needed` a post that already links a paper or repository is left alone, because its replies
have nothing to add. A post whose text says *"link in the reply"* is always read, since that phrasing is
the author stating outright that the root is incomplete.

### Two passes over a batch

`--two-pass` (the default for bookmark imports) captures the whole list publicly first — free, seconds
per source — then reads threads only for the posts that came back without one. Browser time is spent on
posts that demonstrably need it rather than on the whole list, and the second pass upgrades the record
the first stored, so each source appears once at its best captured state.

Upgrading is keyed on what a capture *reached*, not which reader produced it, so an API thread replaces a
root-post-only record exactly as a browser capture does.

The **bookmarklet** is not a capture method; it supplies a list of URLs the other surfaces then read.
Drag it from the Add page, click it on `x.com/i/bookmarks`, and upload the file it saves. It runs in the
user's own browser, in their session, only when clicked — nothing automates a signed-in session to get
the list. `steering import-bookmarks <file>` does the same from the command line.

The **X API** is the only billed surface and stays inert until `STEERING_X_API_CLIENT_ID` is set and the
user authorizes it. Authorization is OAuth 2.0 with PKCE and no client secret, because STEERING runs
where a secret could not be kept; `offline.access` is requested so an unattended run does not need a
human again two hours later. The token lives in the OS keyring, never in the JSON configuration. Its
reply search reaches back seven days, so an older thread returns its root post rather than failing.

### Unattended runs

```powershell
# once, interactive: a human types the credentials, so this step is always visible
# then, unattended:
uv run steering add --batch saved-links.txt --bundle-threads
```

Set `STEERING_BROWSER_HEADLESS=true` in `.env.local` (or `browser_headless` in `config.json`) to drop the window
entirely for captures. Sign-in ignores that setting and stays visible, because a person has to type into it.

A batch reports what it captured and what it did not: a dead link is recorded and the run continues, so one bad
source cannot discard an overnight run. A missing sign-in session under `always` is the one exception and stops
the run immediately, since it would otherwise degrade every remaining item invisibly. Under `auto` a missing
session simply means no escalation, so the run continues on public captures.

## Resolver contract

Implementations provide canonical matching, capture, provenance locators, and explicit availability/error states.
Every contribution needs fixtures and tests for:

- canonical and alternate URLs;
- successful extraction;
- unavailable or private content;
- malformed input and unsafe redirects;
- exact provenance locators;
- prompt-injection content;
- platform and credential limitations.

Fixtures must be synthetic or redistributable. Never commit cookies, tokens, private posts, or normal browser
profiles. Live platform tests are manual or scheduled and are not required for contributor pull requests.
