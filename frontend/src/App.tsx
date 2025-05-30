import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext, {
  ImageLoadInfo,
  AnalysisInfo,
  IR_LIMIT_PX,
} from "./components/interfaces";

import Topbar from "./components/Topbar";
import DragDrop from "./components/DragDrop";
import PreviewCanvas from "./components/Canvas";
import NormalSlider from "./components/NormalSlider";
import { Menu } from "./components/Menu";
import { ErrorMessage, CLSModal, MoreInfo } from "./components/Popups";

import {
  loadFromTIFF,
  loadFromImage,
  getPhaseFraction,
  mean,
} from "./components/imageLogic";

import "./assets/scss/App.scss";
import "bootstrap/dist/css/bootstrap.min.css";

const PATH = "http://127.0.0.1:5000";
// const PATH = "https://samba-segment.azurewebsites.net";
//const PATH = "http://localhost:7071/api";
//const PATH = "https://representative.azurewebsites.net/api"
const PF_ENDPOINT = PATH + "/phasefraction";
//const PF_ENDPOINT = "https://representativity.azurewebsites.net/phasefraction"

//const REPR_ENDPOINT = PATH + "/repr";
const REPR_ENDPOINT = PATH + "/repr";

const MAX_FILE_SIZE_BYTES = 1024 * 1024 * 500; // 500MB

// TODO:
// put phase fracs in image load info
// add isAccurate flag if its not frontend
// make frontend average pf over imageInfos
const App = () => {
  const {
    imageInfo: [imageInfo, setImageInfo],
    selectedImgIdx: [selectedImgIdx, setSelectedImgIdx],
    previewImg: [previewImg, setPreviewImg],
    selectedPhase: [selectedPhase, setSelectedPhase],
    selectedConf: [selectedConf, setSelectedConf],
    errVF: [errVF, setErrVF],
    targetL: [targetL, setTargetL],
    pfB: [, setPfB],
    analysisInfo: [, setAnalysisInfo],
    menuState: [menuState, setMenuState],
    errorState: [errorState, setErrorState],
    showWarning: [showWarning, setShowWarning],
  } = useContext(AppContext)!;

  const allImageInfos = useRef<ImageLoadInfo[]>([]);
  const [nImgs, setNImgs] = useState<number>(0);

  const appLoadFile = async (file: File) => {
    const reader = new FileReader();
    const extension = file.name.split(".").pop()?.toLowerCase();

    const isTIF = extension === "tif" || extension === "tiff";
    const isPNGJPG =
      extension === "png" || extension === "jpg" || extension === "jpeg";
    // need to parse file as arr buffer for tiff but href for png/jpeg etc.
    if (isTIF) {
      reader.readAsArrayBuffer(file);
    } else if (isPNGJPG) {
      reader.readAsDataURL(file);
    } else {
      setErrorState({
        msg: `Unsupported file format .${extension}`,
        stackTrace: "",
      });
      return;
    }

    reader.onload = async () => {
      let result: ImageLoadInfo | null = null;
      if (file.size > MAX_FILE_SIZE_BYTES) {
        setErrorState({
          msg: `File too large!`,
          stackTrace: `File .${file.size / (1000 * 1000)}MB greater than max size (500MB)`,
        });
        return;
      }

      if (isTIF) {
        result = loadFromTIFF(reader.result as ArrayBuffer);
      } else if (isPNGJPG) {
        const href = reader.result as string;
        result = await loadFromImage(href);
      }
      console.log("uploaded file: ", result);

      if (result == null) {
        setErrorState({
          msg: "Failed to load data",
          stackTrace: "",
        });
        return;
      }

      if (result.segmented == false) {
        setErrorState({
          msg: "Data is unsegmented - try using our web segmentation tool, SAMBA (https://www.sambasegment.com/)",
          stackTrace: "Number of unique values > 6",
        });
        return;
      } else {
        if (
          result.height < 200 ||
          result.width < 200 ||
          (result.depth < 200 && result.depth > 1)
        ) {
          setShowWarning("size");
        }

        requestPhaseFraction(file, result);
        result.file = file;
        setImageInfo(result);
        setPreviewImg(result.previewImg);
        setMenuState("phase");

        const newAllImgInfo = [...allImageInfos.current, result];
        allImageInfos.current = newAllImgInfo;
        setNImgs(newAllImgInfo.length);
        setSelectedImgIdx(newAllImgInfo.length - 1);
      }
    };
  };

  const requestPhaseFraction = async (file: File, imageInfo: ImageLoadInfo) => {
    try {
      const formData = new FormData();
      formData.append("userFile", file);
      const resp = await fetch(PF_ENDPOINT, { method: "POST", body: formData });
      const obj = await resp.json();
      console.log("server PF for '" + imageInfo.file?.name + "': ", obj);
      const fractions = obj["phase_fractions"] as { [val: number]: number };
      imageInfo.phaseFractions = fractions;
      imageInfo.isAccurate = true;
    } catch (e) {
      const error = e as Error;
      setErrorState({
        msg: "Couldn't fetch phase fractions: data wrong or server down.",
        stackTrace: error.toString(),
      });
    }
  };

  const requestRepr = async () => {
    try {
      const info = imageInfo!;

      const formData = new FormData();
      for (let info of allImageInfos.current) {
        formData.append("userFile", info.file!);
      }
      // formData.append("userFile", info.file!);
      formData.append(
        "selected_phase",
        String(info.phaseVals[selectedPhase - 1]),
      );
      formData.append("selected_conf", String(selectedConf));
      formData.append("selected_err", String(errVF));

      const resp = await fetch(REPR_ENDPOINT, {
        method: "POST",
        body: formData,
      });
      const obj = await resp.json();
      console.log("repr result:", obj);

      const absErr: number = obj["abs_err"];

      setMenuState("conf_result_full");
      setAnalysisInfo({
        integralRange: obj["cls"],
        z: 1,
        stdModel: obj["std_model"],
        percentageErr: obj["percent_err"],
        absError: absErr,
        lForDefaultErr: obj["l"],
        vf: 1,
        pf: obj["pf_1d"],
        cumSumSum: obj["cum_sum_sum"],
      });

      const vals = imageInfo?.phaseVals!;
      const phaseFrac = mean(
        allImageInfos.current.map(
          (i) => i.phaseFractions[vals[selectedPhase - 1]],
        ),
      );
      setPfB([phaseFrac - absErr, phaseFrac + absErr]);

      if (obj["cls"] > IR_LIMIT_PX) {
        setShowWarning("cls");
      }
      const minSide = Math.min(imageInfo?.width!, imageInfo?.height!);
      if (obj["l"] < minSide) {
        setShowWarning("over");
      }

      setTargetL(obj["l"]);
    } catch (e) {
      const error = e as Error;
      setErrorState({
        msg: "Couldn't determine representativity: data wrong or server down.",
        stackTrace: error.toString(),
      });
    }
  };

  const reset = () => {
    setMenuState("hidden");
    setPreviewImg(null);
    setImageInfo(null);
    setAnalysisInfo(null);
    setTargetL(null);
    setSelectedPhase(0);
    setErrVF(5);
    setSelectedConf(95);
    setErrorState({ msg: "", stackTrace: "" });
    setShowWarning("");
  };

  const changePhase = () => {
    setMenuState("phase");
    setAnalysisInfo(null);
    setTargetL(null);
    const newPhase = selectedPhase % imageInfo?.nPhases!;
    setSelectedPhase(newPhase + 1);
    setErrVF(5);
    setSelectedConf(95);
    setErrorState({ msg: "", stackTrace: "" });
    setShowWarning("");
  };

  const setSelectedImg = (n: number) => {
    const idx = Math.min(Math.max(n, 0), nImgs - 1);
    const newImg = allImageInfos.current[idx];
    setImageInfo(newImg);
    setPreviewImg(newImg.previewImg);
    setSelectedImgIdx(idx);
    setSelectedPhase(selectedPhase);
  };

  useEffect(() => {
    if (menuState === "processing") {
      requestRepr();
    }
  }, [menuState]);

  return (
    <div className={`w-full h-full`}>
      <Topbar
        loadFromFile={appLoadFile}
        reset={reset}
        changePhase={changePhase}
        selectedImgIdx={selectedImgIdx}
        setSelectedImgIdx={setSelectedImg}
        nImgs={nImgs}
      ></Topbar>
      <div className={`flex`} style={{ margin: "1.5%" }}>
        {" "}
        {/*Canvas div on left, sidebar on right*/}
        {!previewImg && <DragDrop loadFromFile={appLoadFile} />}
        {previewImg && <PreviewCanvas />}
        {false && <NormalSlider allImageInfos={allImageInfos.current} />}
      </div>
      <Menu allImageInfos={allImageInfos.current} />
      <ErrorMessage />
      {showWarning != "" && <CLSModal />}
      <MoreInfo />
    </div>
  );
};

export default App;
