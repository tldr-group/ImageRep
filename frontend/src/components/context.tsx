import React, { useState } from "react";
import AppContext from "./interfaces";

const AppContextProvider = (props: {
    children: React.ReactElement<any, string | React.JSXElementConstructor<any>>;
}) => {

    const [image, setImage] = useState<HTMLImageElement | null>(null);
    const [userFile, setUserFile] = useState<File | null>(null);

    return (
        <AppContext.Provider
            value={{
                image: [image, setImage],
                userFile: [userFile, setUserFile]
            }}
        >
            {props.children}
        </AppContext.Provider>
    );
}

export default AppContextProvider;