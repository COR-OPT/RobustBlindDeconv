"""
A module containing various utilities for handling RGB images.
"""
module ImageUtils

	using ColorTypes
	using Images

	# convert RGB element to a 3 element array
	rgb2features(elm) = [elm.r, elm.g, elm.b]

	# scale an image taking into account small deviations from the [0, 1] rng.
	scale_img(x) = begin
		if abs(minimum(x)) > abs(maximum(x))
			x = -x  # flip image
		end
		return scaleminmax(0, 1).(x)
	end

	"""
		gray2features(img::Array, tofloat=true)

	Converts a grayscale image of dimension `d x n` to a 1-dimensional array of
	size ``d \\cdot n``.
	If the flag `tofloat` has been set, converts the image features to
	a `Float64` vector.
	"""
	function gray2features(img::Array, tofloat=true)
		imgx, imgy = size(img)
		if tofloat
			return Float64.(reshape(img, imgx * imgy))
		else
			return reshape(img, imgx * imgy)
		end
	end

	"""
		features2gray(img::Array, img_x::Int, img_y::Int)

	Converts a one dimensional array of size `img_x * img_y` into a
	grayscale image of size `img_x x img_y`, after normalizing to the
	range [0, 1].
	"""
	function features2gray(img_sig::Array, img_x, img_y)
		return N0f8.(scale_img(reshape(img_sig, img_x, img_y)))
	end

	"""
		function rgbim2features(img::Array, tofloat=true)

	Converts an RGB image of dimension `d x n` to a 1-dimensional array of
	size `3dn` after "flattening" the image to a vector and then converting
	every RGB element to a consecutive triple of individual elements.
	If the flag `tofloat` has been set, converts the image features to
	a `Float64` vector.
	"""
	function rgbim2features(img::Array, tofloat=true)
		imgx, imgy = size(img)
		img_vec = map(rgb2features, reshape(img, imgx * imgy))
		if tofloat
			return Float64.(vcat(img_vec...)), imgx, imgy
		else
			return vcat(img_vec...), imgx, imgy
		end
	end

	"""
		function rgbim2channelfeatures(img::Array, tofloat=true)

	Converts an RGB image of dimension `d x n` to a 3 1-dimensional arrays of
	size `dn` after converting every channel to a vector.
	"""
	function rgbim2channelfeatures(img::Array, tofloat=true)
		imgx, imgy = size(img)
		convapp = (x -> if tofloat Float64.(x) else x end)
		imgr = reshape(convapp.(red.(img)), imgx * imgy)
		imgg = reshape(convapp.(green.(img)), imgx * imgy)
		imgb = reshape(convapp.(blue.(img)), imgx * imgy)
		return imgr, imgg, imgb, imgx, imgy
	end

	"""
		function features2rgbim(img::Array{Any, 1}, imgx::Int, imgy::Int)

	Convert a `3 x imgx x imgy` vector of values to an RGB image of size
	`imgx x imgy`. Note that, given an image `s`, the following should hold:

	`(features2rgbim(rgbim2features(s)...) == s) == true`.
	"""
	function features2rgbim(img::Array, imgx::Int, imgy::Int)
		# create a 1x[imgx]x[imgy] array of RGB elements
		return reshape(
			mapslices(x -> ColorTypes.RGB(x[1], x[2], x[3]),
			reshape(img, 3, imgx, imgy), 1), imgx, imgy)
	end

	"""
		function rescale(img::Array)

	Rescale an RGB image from 0 to 1.0. This function will isolate the color
	channels, apply a scaling to [0, 1] on each, and recombine them into an
	RGB image.
	"""
	function rescale(imgr::Array, imgg::Array, imgb::Array, szx::Int, szy::Int)
		# scale an image channel
		imgscale(x) = begin
			@show minimum(x), maximum(x)
			x = x .- minimum(x); x = x ./ maximum(x)
			return x
		end
		imgr = imgscale(reshape(imgr, szx, szy))
		imgg = imgscale(reshape(imgg, szx, szy))
		imgb = imgscale(reshape(imgb, szx, szy))
		colorings = reshape(hcat(imgr, imgg, imgb), szx, szy, 3)
		return permutedims(colorings, (3, 1, 2))
	end

	"""
		function rescale_from_chan(imgr::Array, imgg::Array, imgb::Array,
								   szx::Int, szy::Int)

	Rescale an RGB image from 0 to 1.0, given its color channels separately.
	This function will apply a scaling to [0, 1] on each, and recombine them
	into an RGB image.
	"""
	function rescale_from_chan(imgr::Array, imgg::Array, imgb::Array,
							   szx::Int, szy::Int)
		# reshape images
		imgr = scale_img(reshape(imgr, szx, szy))
		imgg = scale_img(reshape(imgg, szx, szy))
		imgb = scale_img(reshape(imgb, szx, szy))
		colorings = reshape(N0f8.(hcat(imgr, imgg, imgb)), szx, szy, 3)
		return Array(ColorView{RGB}(permutedims(colorings, (3, 1, 2))))
	end
end
