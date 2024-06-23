import React, { useState } from "react";
import AppContext, { MenuState } from "./interfaces";

const AppContextProvider = (props: {
    children: React.ReactElement<any, string | React.JSXElementConstructor<any>>;
}) => {

    const [previewData, setPreviewData] = useState<ImageData | null>(null);
    const [previewImg, setPreviewImg] = useState<HTMLImageElement | null>(null);
    const [userFile, setUserFile] = useState<File | null>(null);
    const [selectedPhase, setSelectedPhase] = useState<number>(1);
    const [menuState, setMenuState] = useState<MenuState>('hidden');


    return (
        <AppContext.Provider
            value={{
                previewData: [previewData, setPreviewData],
                previewImg: [previewImg, setPreviewImg],
                userFile: [userFile, setUserFile],
                selectedPhase: [selectedPhase, setSelectedPhase],
                menuState: [menuState, setMenuState]
            }}
        >
            {props.children}
        </AppContext.Provider>
    );
}

export default AppContextProvider;