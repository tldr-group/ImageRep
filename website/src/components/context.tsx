import React, { useState } from "react";
import AppContext from "./createContext";

const AppContextProvider = (props: {
    children: React.ReactElement<any, string | React.JSXElementConstructor<any>>;
}) => {

    return (
        <AppContext.Provider
            value={{}}
        >
            {props.children}
        </AppContext.Provider>
    );
}

export default AppContextProvider;