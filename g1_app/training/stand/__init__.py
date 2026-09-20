"""In-place balance (stand-still) training for legged robots.

Starts every episode from the nominal standing pose (the "walking zero"
position: upright, root at standing height, joints at default) and learns
to stay balanced under strong shoves. Stepping to catch balance is allowed.
"""
