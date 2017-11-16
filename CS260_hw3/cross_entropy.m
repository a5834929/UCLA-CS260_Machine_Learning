function entropy = cross_entropy( label, sig, w, lambda )

err = label.*log(sig) + (1-label).*log(1-sig);
entropy = -sum(err)+lambda*norm(w);

end

