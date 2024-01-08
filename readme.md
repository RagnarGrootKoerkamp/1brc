# One Billion Row Challenge

See [here](https://www.morling.dev/blog/one-billion-row-challenge/) for the
original description of this challenge.

I wrote a blog post explaining my ideas at https://curiouscoding.nl/posts/1brc .

Note that I do make some assumptions:
- Lines in the input are at most 33 characters, including the newline.
- City names are uniquely determined by their first and last 8 characters.

On my `i7-10750H CPU` running at `4.6GHz`, this results in:
- 7.5s wall time on a single thread.
- 1.43s wall time on 6 threads.


