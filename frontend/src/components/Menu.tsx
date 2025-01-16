import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext, { IR_LIMIT_PX, MenuState } from "./interfaces";
import { colours, rgbaToHex } from "./interfaces";

import Toast from 'react-bootstrap/Toast'
import ToastContainer from "react-bootstrap/ToastContainer";
import Button from 'react-bootstrap/Button';
import ButtonGroup from "react-bootstrap/ButtonGroup";
import InputGroup from "react-bootstrap/InputGroup";
import Form from "react-bootstrap/Form";
import Spinner from "react-bootstrap/Spinner";
import Table from "react-bootstrap/Table";
import Modal from 'react-bootstrap/Modal';
import NormalSlider from "./NormalSlider";
import ListGroup from 'react-bootstrap/ListGroup';

import Accordion from 'react-bootstrap/Accordion';
import { getPhaseFraction } from "./imageLogic";

const centreStyle = { display: 'flex', justifyContent: 'space-evenly', alignItems: 'center', marginTop: '1em' }

const _getCSSColour = (currentStateVal: any, targetStateVal: any, successPrefix: string, colourIdx: number): string => {
    // Boring function to map a success to current labelling colour. Used for GUI elements.
    const c = colours[colourIdx];
    const hex = rgbaToHex(c[0], c[1], c[2], 255);
    const matches: boolean = (currentStateVal === targetStateVal);

    const outlineStr = (matches) ? successPrefix + hex : 'white'
    return outlineStr;
}

const restyleAccordionHeaders = (ref: React.RefObject<HTMLHeadingElement>, primary: boolean, hex: string) => {
    const headerBGCSSName = "--bs-accordion-active-bg"
    const headerTextCSSName = "--bs-accordion-active-color"
    const headerDefaultBGCSSName = "--bs-accordion-bg"
    const header = ref.current

    const colour = (primary) ? hex : "#ffffff"

    header?.style.setProperty(headerBGCSSName, colour)
    header?.style.setProperty(headerTextCSSName, "#212529")

    if (primary) {
        header?.style.setProperty(headerDefaultBGCSSName, colour)
        header?.style.setProperty('background-color', colour)
    }
    //header?.style.removeProperty("background-image")
}

const PhaseSelect = () => {
    const {
        imageInfo: [imageInfo,],
        selectedPhase: [selectedPhase, setSelectedPhase],
        menuState: [, setMenuState]
    } = useContext(AppContext)!

    const classes: number[] = Array.from(new Array(imageInfo!.nPhases), (_, i) => i + 1);

    const getStyle = (i: number) => {
        return {
            backgroundColor: _getCSSColour(i, selectedPhase, "", selectedPhase),
            border: _getCSSColour(i, i, "2px solid", i),
            margin: '1px 1px 1px 1px'
        }
    }

    const confirm = () => {
        if ((selectedPhase > 0) && (selectedPhase < 7)) {
            setMenuState('conf');
            return;
        } else {
            return;
        }
    }

    return (
        <>
            <p>Choose phase to analyze representativity of:</p>
            <ButtonGroup style={{ paddingLeft: "3%", marginLeft: '0%' }}>
                {
                    classes.map(i => <Button key={i} variant="light" onClick={(e) => setSelectedPhase(i)} style={getStyle(i)}>{i}</Button>)
                }
            </ButtonGroup>
            <Button variant="dark" onClick={(e) => { confirm() }}>Confirm</Button>
        </>
    );
}

const ConfidenceSelect = () => {
    const {
        imageInfo: [imageInfo,],
        selectedPhase: [selectedPhase,],
        selectedConf: [selectedConf, setSelectedConf],
        errVF: [errVF, setErrVF],
        accurateFractions: [accurateFractions,],
        menuState: [, setMenuState]
    } = useContext(AppContext)!


    const vals = imageInfo?.phaseVals!

    console.log(accurateFractions)
    console.log(vals)
    console.log(selectedPhase)

    const displayPhaseFrac = () => {
        try {
        if (accurateFractions == null && imageInfo && vals[selectedPhase -1] && imageInfo?.previewData.data) {
            console.log('help')
            const pf = getPhaseFraction(
                imageInfo?.previewData.data!,
                vals[selectedPhase - 1]
            )
            if (typeof pf == "number") {
                return pf.toFixed(3)
            }
        } else if (accurateFractions && vals[selectedPhase -1]) {
            return accurateFractions[vals[selectedPhase - 1]].toFixed(3)
        }

        return ''
    } catch (e) {
        return ''
    }
    }
    // horrible ternary: if server has responded and set the accurate phase fractions,
    // then use those values in the modal. If not, use the estimate from the first image
    const phaseFrac = displayPhaseFrac()

    const setConf = (e: any) => {
        setSelectedConf(Number(e.target!.value))
    }

    const setErr = (e: any) => {
        setErrVF(Number(e.target!.value))
    }

    const [h, w, d] = [imageInfo?.height!, imageInfo?.width!, imageInfo?.depth!];
    const dimString = (imageInfo?.nDims == 3) ? `${h}x${w}x${d}` : `${h}x${w}`;
    const n = (imageInfo?.nDims == 3) ? h * w * d : h * w
    const fitTimeConstant = 48500
    const t = Math.sqrt(n / fitTimeConstant)


    return (
        <>
            <Table>
                <tbody>
                    <tr>
                        <td>Image Dimensions:</td>
                        <td>{dimString}</td>
                    </tr>
                    <tr>
                        <td >Chosen Phase:</td>
                        <td >{selectedPhase}</td>
                    </tr>
                    <tr>
                        <td>Phase Fraction:</td>
                        <td>{phaseFrac}</td>
                    </tr>
                    <tr>
                        <td>Estimated Time:</td>
                        <td>{t.toFixed(1)}s</td>
                    </tr>
                </tbody>
            </Table>
            <InputGroup>
                <InputGroup.Text>Uncertainty Target (%):</InputGroup.Text>
                <Form.Control type="number" min={0} max={100} value={errVF} onChange={(e) => setErr(e)} width={1} size="sm"></Form.Control>
            </InputGroup>
            <InputGroup>
                <InputGroup.Text>Confidence in Bounds (%):</InputGroup.Text>
                <Form.Control type="number" min={0} max={100} value={selectedConf} onChange={(e) => setConf(e)} width={1} size="sm"></Form.Control>
            </InputGroup>
            <div style={centreStyle}>
                <Button variant="dark" onClick={(e) => { setMenuState('processing') }}>Calculate!</Button>
            </div>
        </>
    )
}

const Result = () => {
    // LB < VF_true < UB with '$CONF$% confidence  
    // need L = $N$pix for \epsilon = ...
    // epsilon slider (updates bounds in line 1)
    // conf re-select
    // CSS zoom anim on canvas
    const {
        analysisInfo: [analysisInfo,],
        imageInfo: [imageInfo,],
        selectedPhase: [selectedPhase,],
        selectedConf: [selectedConf, setSelectedConf],
        errVF: [errVF, setErrVF],
        pfB: [pfB,],
        accurateFractions: [accurateFractions,],
        menuState: [menuState, setMenuState],
        showInfo: [, setShowInfo],
    } = useContext(AppContext)!

    const newConfSliderRef = useRef<HTMLInputElement>(null);

    // we have two errVFs here because we want the values in the text to reflect the old
    // errVF, the one they sent to the server and the slider to represent the new one
    // which they are setting for recalculate.

    // TODO: THERE IS RACE CONDITION BETWEEN ACCURATE PFS AND RESULTS: I.E IF ONE
    // NOT RETURNED WEBSITE FALLS OVER - NOT GOOD. FIX
    // MAY EVEN BE LAG IN COUNTING PHASE FRACS
    const [newErrVF, setNewErrVF] = useState<number>(5);
    const pfResultRef = useRef<HTMLHeadingElement>(null);
    const lResultRef = useRef<HTMLHeadingElement>(null);

    const vals = imageInfo?.phaseVals!

    const getPhaseFracs = () => {
        const accurateAvailable = accurateFractions != null
        const coarseAvailable = (imageInfo != null) && (imageInfo.previewData.data != null)

        if (accurateAvailable) {
            return accurateFractions[vals[selectedPhase - 1]]
        } else if (coarseAvailable) {
            return getPhaseFraction(
                imageInfo?.previewData.data!,
                vals[selectedPhase - 1]
            );
        } else {
            return 0
        }
    }

    const phaseFrac = getPhaseFracs();


    const l = analysisInfo?.lForDefaultErr;
    const lStr = l?.toFixed(0);
    const sizeText = (imageInfo?.nDims == 3) ? `${lStr}x${lStr}x${lStr}` : `${lStr}x${lStr}`

    const setErr = (e: any) => {
        setNewErrVF(Number(e.target!.value));
    };

    const recalculate = () => {
        const newSlider = newConfSliderRef.current!

        const val = newSlider.value
        const isNumber = !isNaN(parseFloat(val));
        const floatVal = parseFloat(val)
        const inBounds = (floatVal > 1) && (floatVal < 99.999)

        if (isNumber && inBounds) {
            setSelectedConf(floatVal)
        } else {
            setSelectedConf(95)
        }
        setErrVF(newErrVF);
        setMenuState('processing');
    }

    const getDPofSigFig = (decimal: number) => {
        const rounded = parseFloat(decimal.toPrecision(1));
        const loc = Math.ceil(Math.abs(Math.log10(rounded)));
        const capped = Math.min(loc, 5)
        return capped
    }

    const c = colours[selectedPhase];
    const headerHex = rgbaToHex(c[0], c[1], c[2], c[3]);

    useEffect(() => {
        const refs = [pfResultRef, lResultRef];
        refs.map((r, i) => restyleAccordionHeaders(r, (i == 0), headerHex));
    }, [selectedPhase])


    const absErrFromPFB = (pfB![1] - pfB![0]) / 2
    const perErrFromPFB = 100 * (((pfB![1] - pfB![0]) / 2) / phaseFrac)

    const roundTo = getDPofSigFig(absErrFromPFB);

    const beforeBoldText = `The phase fraction in the segmented image is ${phaseFrac.toFixed(3)}. Assuming perfect segmentation, the 'ImageRep' model proposed by Dahari et al. suggests that `
    const boldText = `we can be ${selectedConf.toFixed(1)}% confident that the material's phase fraction is within ${perErrFromPFB?.toFixed(1)}% of this value (i.e. ${phaseFrac.toFixed(roundTo)}±${(absErrFromPFB).toFixed(roundTo)})`
    const copyText = beforeBoldText + boldText
    const afterText = "These results are derived from an estimated "
    const afterBoldText = `Characteristic Length Scale (CLS) of ${analysisInfo?.integralRange!.toFixed(0)}px`

    const copyBtn = () => { navigator.clipboard.writeText(copyText) }

    const ii = imageInfo
    const vol = (ii?.nDims! == 3) ? (ii?.height! * ii?.width! * ii?.width!) : (ii?.height! * ii?.width!)
    const nMore = (Math.ceil(Math.pow(l!, imageInfo?.nDims!) / vol)) - 1

    const modalTitle = `Results for "${imageInfo?.file?.name}"`

    const title = "Phase Fraction Estimation of the Material"

    const smallResults = (
        <>
            <Accordion defaultActiveKey={['0', '1']} flush alwaysOpen>
                <Accordion.Item eventKey="0">
                    <Accordion.Header ref={pfResultRef}>{title}</Accordion.Header>
                    {/*Need to manually overwrite the style here because of werid bug*/}
                    <Accordion.Body style={{ visibility: "visible" }}>
                        {beforeBoldText}<b>{boldText}</b>
                        <InputGroup style={{ justifyContent: 'center', marginTop: '1em' }}>
                            <InputGroup.Text id="btnGroupAddon">Copy:</InputGroup.Text>
                            <Button variant="outline-secondary" onClick={copyBtn}>text</Button>
                            <Button variant="outline-secondary">citation</Button>
                        </InputGroup>
                    </Accordion.Body>
                </Accordion.Item>
                <Accordion.Item eventKey="1" >
                    <Accordion.Header ref={lResultRef}>Required Length for Target</Accordion.Header>
                    {/*Need to manually overwrite the style here because of werid bug*/}
                    <Accordion.Body style={{ visibility: "visible" }}>
                        For a {errVF.toFixed(2)}% uncertainty in phase fraction, you <b>need to measure a total image size of about {sizeText} (i.e. {nMore} more images)</b> at the same resolution.
                    </Accordion.Body>
                </Accordion.Item>
            </Accordion >

            {/*
            <p><b>&nbsp;&nbsp;&nbsp;&nbsp; {targLB.toFixed(3)} ≤ ϕ ≤ {targUB.toFixed(3)} with {selectedConf}% confidence.</b></p>
            */}
            <p></p>
            < InputGroup >
                <InputGroup.Text>Uncertainty Target (%):</InputGroup.Text>
                <Form.Control type="number" min={0} max={100} value={newErrVF} onChange={(e) => setErr(e)} width={1} size="sm"></Form.Control>
            </InputGroup >
            <InputGroup>
                <InputGroup.Text>Confidence in Bounds (%):</InputGroup.Text>
                <Form.Control ref={newConfSliderRef} type="number" min={0} max={100} defaultValue={selectedConf} width={1} size="sm"></Form.Control>
            </InputGroup>
            <div style={centreStyle}>
                <Button variant="outline-dark" onClick={(e) => { setShowInfo(true) }}>More Info</Button>
                <Button variant="dark" onClick={(e) => { recalculate() }}>Recalculate!</Button>
            </div>
        </>
    )

    const handleClose = () => { setMenuState('conf_result') };
    const handleShowFull = () => {
        setMenuState('hidden');
        setShowInfo(true);
    };

    const showFull = (menuState == 'conf_result_full')

    const largeResults = (<>
        <Modal show={showFull} onHide={handleClose} size="lg">
            <Modal.Header style={{ backgroundColor: '#212529', color: '#ffffff' }} closeVariant="white" closeButton>
                <Modal.Title>{modalTitle}</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <Accordion defaultActiveKey={['0', '1']} flush alwaysOpen>
                    <Accordion.Item eventKey="0">
                        <Accordion.Header ref={pfResultRef}>{title}</Accordion.Header>
                        {/*Need to manually overwrite the style here because of werid bug*/}
                        <Accordion.Body style={{ visibility: "visible" }}>
                            <div style={{ display: 'flex', alignItems: 'center' }}>
                                <NormalSlider ></NormalSlider>
                            </div>
                            {beforeBoldText}<b>{boldText}</b>
                            <InputGroup style={{ justifyContent: 'center', marginTop: '1em', marginBottom: '1em' }}>
                                <InputGroup.Text id="btnGroupAddon">Copy:</InputGroup.Text>
                                <Button variant="outline-secondary" onClick={copyBtn}>text</Button>
                                <Button variant="outline-secondary">citation</Button>
                            </InputGroup>
                            {afterText}<b>{afterBoldText}</b>
                        </Accordion.Body>
                    </Accordion.Item>
                    <Accordion.Item eventKey="1" >
                        <Accordion.Header ref={lResultRef}>Required Length for Target</Accordion.Header>
                        {/*Need to manually overwrite the style here because of werid bug*/}
                        <Accordion.Body style={{ visibility: "visible" }}>
                            For a {errVF.toFixed(2)}% uncertainty in phase fraction, you <b>need to measure a total image size of about {sizeText} (i.e. {nMore} more images)</b> at the same resolution.
                            <div style={{ display: 'flex', justifyContent: 'flex-end', }}>
                                <Button variant="dark" onClick={handleClose}>
                                    Visualise!
                                </Button>
                            </div>

                        </Accordion.Body>
                    </Accordion.Item>
                    <Accordion.Item eventKey="2" >
                        <Accordion.Header style={{ color: 'red' }}>Warnings</Accordion.Header>
                        <Accordion.Body style={{ visibility: "visible" }}>
                            <ListGroup>
                                <ListGroup.Item>Other errors are possible, and may be larger! (i.e, segmentation error)</ListGroup.Item>
                                <ListGroup.Item>Not designed for periodic structures</ListGroup.Item>
                                <ListGroup.Item>This is a (conservative) estimate - retry when you have measured the larger sample</ListGroup.Item>
                            </ListGroup>
                        </Accordion.Body>
                    </Accordion.Item>
                    <Accordion.Item eventKey="3" >
                        <Accordion.Header style={{ color: 'red' }}>More info</Accordion.Header>
                        <Accordion.Body style={{ visibility: "visible" }}>
                            <ListGroup>
                                <ListGroup.Item variant="dark" style={{ cursor: "pointer" }} onClick={e => handleShowFull()}>Click for Brief explanation!</ListGroup.Item>
                                <ListGroup.Item>Implementation in the <a href="https://github.com/tldr-group/Representativity">GitHub</a></ListGroup.Item>
                                <ListGroup.Item>Full details can be found in the <a href="https://arxiv.org/abs/2410.19568v1">paper</a></ListGroup.Item>
                            </ListGroup>
                        </Accordion.Body>
                    </Accordion.Item>
                </Accordion >
            </Modal.Body>
            <Modal.Footer>
                <Button variant="dark" onClick={handleClose}>
                    Understood!
                </Button>
            </Modal.Footer>
        </Modal >
    </>)

    return (<>
        {(showFull == true) && largeResults}
        {(showFull == false) && smallResults}
    </>)
}


const getMenuInfo = (state: MenuState) => {
    switch (state) {
        case 'phase':
            return { title: "Select Phase", innerHTML: <PhaseSelect /> }
        case 'conf':
            return { title: "Choose Parameters", innerHTML: <ConfidenceSelect /> }
        case 'processing':
            return { title: "Processing", innerHTML: <div style={centreStyle}><Spinner /></div> }
        case 'conf_result_full':
            return { title: "Results", innerHTML: <Result /> }
        case 'conf_result':
            return { title: "Results", innerHTML: <Result /> }
        case 'hidden': // fall through
        default:
            return { title: "", innerHTML: <></> }
    }
}

export const Menu = () => {
    const {
        menuState: [menuState,],
    } = useContext(AppContext)!

    const [collapse, setCollapse] = useState<boolean>(false);

    const hide = (menuState == 'hidden')

    return (
        <>
            {(menuState == 'conf_result_full') ? getMenuInfo(menuState).innerHTML :
                <ToastContainer className="p-5" position="bottom-end" >
                    <Toast show={!hide}>
                        <Toast.Header className="roundedme-2" closeButton={false}>
                            <strong className="me-auto" style={{ fontSize: '1.5em' }}>{getMenuInfo(menuState).title}</strong>
                            <Button onClick={(e) => setCollapse(!collapse)} variant="outline-dark" size="sm">{collapse ? `▼` : `▲`}</Button>
                        </Toast.Header>
                        <Toast.Body>
                            {(collapse == false) && getMenuInfo(menuState).innerHTML}
                        </Toast.Body>
                    </Toast>
                </ToastContainer>}

        </>
    )
}
