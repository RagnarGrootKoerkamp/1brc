# One Billion Row Challenge

See [here](https://www.morling.dev/blog/one-billion-row-challenge/) for the
original description of this challenge.

I wrote a blog post explaining my ideas at https://curiouscoding.nl/posts/1brc .

Note that I do make some assumptions:
- Lines in the input are at most 33 characters, including the newline.
- City names are uniquely determined by their first and last 8 characters.
- Each input line contains a city drawn uniform random from the set of available cities.

On my `i7-10750H CPU` running at `4.6GHz`, this results in:
- 5.72s wall time on a single thread.
- 1.01s wall time on 12 threads on 6 cores.


## While you're here
The `justfile` contains some often used commands that can be run using [`just`](https://github.com/casey/just).

For this project, I wrote a small shell
[`just-shell`](https://github.com/RagnarGrootKoerkamp/just-shell) to
conveniently run `just` commands. It has super aggressive auto-completion, which
makes running a small set of commands very convenient.
