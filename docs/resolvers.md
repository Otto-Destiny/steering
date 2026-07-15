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
