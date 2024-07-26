import React, { useState } from "react";
import AppContext, { AnalysisInfo, ImageLoadInfo, MenuState, ErrorMessage } from "./interfaces";

const AppContextProvider = (props: {
    children: React.ReactElement<any, string | React.JSXElementConstructor<any>>;
}) => {
    // user files
    const [imageInfo, setImageInfo] = useState<ImageLoadInfo | null>(null);
    const [previewImg, setPreviewImg] = useState<HTMLImageElement | null>(null);
    // user options
    const [selectedPhase, setSelectedPhase] = useState<number>(0);
    const [selectedConf, setSelectedConf] = useState<number>(95);
    const [errVF, setErrVF] = useState<number>(5);
    const [targetL, setTargetL] = useState<number | null>(null);
    // server data
    const [accurateFractions, setAccurateFractions] = useState<{ [val: number]: number } | null>(null);
    const [analysisInfo, setAnalysisInfo] = useState<AnalysisInfo | null>(null);
    // control flow
    const [menuState, setMenuState] = useState<MenuState>('hidden');
    const [errorState, setErrorState] = useState<ErrorMessage>({ msg: "", stackTrace: "" });
    const [showWarning, setShowWarning] = useState<"" | "cls" | "size" | "over">("");
    const [showInfo, setShowInfo] = useState<boolean>(false);
    const [showFullResults, setShowFullResults] = useState<boolean>(false);


    return (
        <AppContext.Provider
            value={{
                imageInfo: [imageInfo, setImageInfo],
                previewImg: [previewImg, setPreviewImg],
                selectedPhase: [selectedPhase, setSelectedPhase],
                selectedConf: [selectedConf, setSelectedConf],
                errVF: [errVF, setErrVF],
                targetL: [targetL, setTargetL],
                accurateFractions: [accurateFractions, setAccurateFractions],
                analysisInfo: [analysisInfo, setAnalysisInfo],
                menuState: [menuState, setMenuState],
                errorState: [errorState, setErrorState],
                showWarning: [showWarning, setShowWarning],
                showInfo: [showInfo, setShowInfo],
                showFullResults: [showFullResults, setShowFullResults],
            }}
        >
            {props.children}
        </AppContext.Provider>
    );
}

export default AppContextProvider;