import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext from "./components/interfaces";

import Topbar from "./components/Topbar"
import DragDrop from "./components/DragDrop";

import "./assets/scss/App.scss";
import 'bootstrap/dist/css/bootstrap.min.css';



const App = () => {
    const {
        image: [image, setImage],
    } = useContext(AppContext)!

    const foo = () => { }
    const fileFoo = (file: File) => { console.log('file uploaded') }

    useEffect(() => { }, [])

    return (
        <div className={`w-full h-full`}>
            <Topbar></Topbar>
            <div className={`flex`} style={{ margin: '1.5%' }} > {/*Canvas div on left, sidebar on right*/}
                {!image && <DragDrop loadDefault={foo} loadFromFile={fileFoo} />}
            </div>
        </div>
    );
};

export default App;