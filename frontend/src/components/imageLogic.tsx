import { ImageLoadInfo } from "./interfaces";

const UTIF = require("./UTIF.js");

type PhaseCheck = { nPhases: number; segmented: boolean; vals: number[] };
const checkPhases = (
  arr: Uint8ClampedArray,
  nChannels: number = 4,
): PhaseCheck => {
  // given 2D image data, find N unique values, if > 6 call it unsegmented
  // ASSUMES RGBA ARRAYS: valid as UTIF decodes to RGBA
  const uniqueColours = arr.filter((_, i, __) => {
    return i % nChannels == 0;
  });
  const uniqueValues = [...new Set(uniqueColours)].sort((a, b) => a - b); // create Set (only unique vals) then unpack into arr
  const nPhases = uniqueValues.length;
  const segmented = nPhases < 6 ? true : false;
  // TODO: if data RGBA then opacity will be counted as phase - this is bug
  // need to explictly look for unique rgb values
  return { nPhases: nPhases, segmented: segmented, vals: uniqueValues };
};

const findNChannels = (arr: Uint8ClampedArray, ih: number, iw: number) => {
  return Math.round(arr.length / (ih * iw));
};

export const replaceGreyscaleWithColours = (
  arr: Uint8ClampedArray,
  mapping: { [greyVal: number]: Array<number> },
  nChannels: number = 4,
) => {
  const nPixels = Math.floor(arr.length / nChannels);
  const out = new Uint8ClampedArray(nPixels * 4).fill(0);
  for (let i = 0; i < nPixels; i = i + 1) {
    const queryVal = arr[nChannels * i];
    if (queryVal in mapping) {
      const [R, G, B, A] = mapping[queryVal];
      out[4 * i] = R;
      out[4 * i + 1] = G;
      out[4 * i + 2] = B;
      out[4 * i + 3] = A;
    }
  }
  return out;
};

export const getPhaseFraction = (
  arr: Uint8ClampedArray,
  val: number,
  nChannels: number = 4,
) => {
  const uniqueVals = arr.filter((_, i, __) => {
    return i % nChannels == 0;
  });
  const matching = uniqueVals.filter((v) => v == val);
  if (arr.length == 0) {
    return 0;
  }
  return matching.length / (arr.length / nChannels);
};

const imageInfoLoadHelper = (
  arr: ImageData,
  img: HTMLImageElement,
  nDims: 2 | 3,
  phaseCheck: PhaseCheck,
  h: number,
  w: number,
  d: number,
): ImageLoadInfo => {
  const phaseFracs = Object.fromEntries(
    phaseCheck.vals.map((v) => [v, getPhaseFraction(arr.data, v)]),
  );

  return {
    file: null,
    previewData: arr,
    previewImg: img,
    nDims: nDims,
    nPhases: phaseCheck.nPhases,
    phaseVals: phaseCheck.vals,
    segmented: phaseCheck.segmented,
    height: h,
    width: w,
    depth: d,
    phaseFractions: phaseFracs,
    isAccurate: false,
  };
};

export const loadFromTIFF = (tiffBuffer: ArrayBuffer): ImageLoadInfo => {
  const tifs: Array<any> = UTIF.decode(tiffBuffer);
  const tif = tifs[0];
  // this needs to be done in-place before we can read the data
  UTIF.decodeImage(tiffBuffer, tif);

  const imgDataArr = new Uint8ClampedArray(UTIF.toRGBA8(tif));
  const imgData = new ImageData(imgDataArr, tif.width, tif.height);
  const img = getImagefromImageData(imgData, tif.height, tif.width);

  const nDims = tifs.length > 1 ? 3 : 2;

  const phaseCheck = checkPhases(imgDataArr);

  return imageInfoLoadHelper(
    imgData,
    img,
    nDims,
    phaseCheck,
    tif.height,
    tif.width,
    tifs.length,
  );
};

const getImageDataFromImage = (image: HTMLImageElement): ImageData => {
  // create temp canvas, draw image, get image data from canvas and return
  const tmpCanvas = document.createElement("canvas");
  const tmpContext = tmpCanvas.getContext("2d")!;

  tmpCanvas.width = image.width;
  tmpCanvas.height = image.height;
  tmpContext.drawImage(image, 0, 0);

  const data = tmpContext.getImageData(0, 0, image.width, image.height);
  tmpCanvas.remove();
  return data;
};

export const getImagefromImageData = (
  imageData: ImageData,
  height: number,
  width: number,
): HTMLImageElement => {
  const tmpCanvas = document.createElement("canvas");
  const tmpContext = tmpCanvas.getContext("2d")!;

  tmpCanvas.height = height;
  tmpCanvas.width = width;
  tmpContext.putImageData(imageData, 0, 0);

  const img = new Image(width, height);
  img.src = tmpCanvas.toDataURL();
  tmpCanvas.remove();
  return img;
};

export const loadFromImage = async (href: string): Promise<ImageLoadInfo> => {
  // load href to Image element, draw to temp canvas to extract pixel data
  const img = new Image();
  img.src = href;
  // decode ~ 'promisified onload handler' i.e blocks until source loaded
  await img.decode();

  const imgData = getImageDataFromImage(img);
  const nChannels = findNChannels(imgData.data, img.height, img.width);
  const phaseCheck = checkPhases(imgData.data, nChannels);

  return imageInfoLoadHelper(
    imgData,
    img,
    2,
    phaseCheck,
    img.height,
    img.width,
    1,
  );
};

export const mean = (arr: number[]) => {
  return (
    arr.reduce((acc, v) => {
      return acc + v;
    }) / arr.length
  );
};

export const getNImagesForTargetL = (
  imageInfo: ImageLoadInfo,
  l: number,
  to_subtract: number = 1,
) => {
  const ii = imageInfo;
  const vol =
    ii?.nDims! == 3
      ? ii?.height! * ii?.width! * ii?.width!
      : ii?.height! * ii?.width!;
  const nMore = Math.ceil(Math.pow(l!, imageInfo?.nDims!) / vol) - to_subtract;
  return nMore;
};
