Addresses issue # .

**Is this a bugfix or an enhancement?**

**Proposed changes:**







**Checklist:**
The below is a list of things what will be considered when reviewing PRs. It is not prescriptive, and does not
imply that PRs which touch any of these will be rejected but gives a rough indication of where there is a potential 
for hangups (i.e. factors which could turn a 5 min review into a half hour or longer and shunt it to the bottom
of the TODO list).

- [ ] Does the PR avoid variable renaming in existing code, whitespace changes, and other forms of tidying? [There is a place for code tidying, but it makes reviewing 
much simpler if this is kept separate from functional changes. The auto-formatting performed by some editors is particulaly egregious and can lead to files with thousands
of non-functional changes with a few functional changes scattered amoungst them]

If an enhancement (or non-trivial bugfix):

- [ ] Has this been discussed in advance (feature request, PR proposal, email, or direct conversation)?
- [ ] Does this change how users interact with the software? How will these changes be communicated?
- [ ] Does this maintain backwards compatibility with old data?
- [ ] Does this change the required dependencies?
- [ ] Are there any other side effects of the change?
